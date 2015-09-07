"""An implementation of Strata

Based on http://www.eng.yale.edu/wenjun/papers/strata.pdf"""

import collections
import itertools
import random

import numpy as np

TOP = 0
BOTTOM = 3
LEFT = 1
RIGHT = 2

NLAYERS = 4


def count_pixels(img):
    """Returns the number of black and white pixels"""
    nblack = np.sum(img[:, :] == 0)
    nwhite = np.sum(img[:, :] == 1)
    return nblack, nwhite


def get_subblock_imgs(img, orientation, reserved=False):
    """Returns 8 sub-blocks of `img`

    For example, if orientation is LEFT and img looks like:

        x x A B
        x x C D
        x x E F
        x x G H,

    then this function returns tuple(A, B, C, D, E, F, G, H)"""
    n2 = img.shape[0]/2
    n4 = img.shape[0]/4

    if not reserved:
        # Flip the orientation
        orientation = (~orientation) & 0x3

    quarters = (slice(0, n4),      slice(n4, 2*n4),
                slice(2*n4, 3*n4), slice(3*n4, None))
    upper_halves = (slice(0,    n4), slice(n4,   n2))
    lower_halves = (slice(n2, 3*n4), slice(3*n4, None))

    if orientation == TOP:
        slices = itertools.product(upper_halves, quarters)
    elif orientation == BOTTOM:
        slices = itertools.product(lower_halves, quarters)
    elif orientation == LEFT:
        slices = itertools.product(quarters, upper_halves)
    else:
        if orientation != RIGHT:
            print orientation, reserved
        assert orientation == RIGHT
        slices = itertools.product(quarters, lower_halves)

    return tuple(img[slice_] for slice_ in slices)


def get_extra_strip_imgs(img, orientation, layer):
    """Return the layer-4 blocks located in extra strips."""
    subimgs = get_subblock_imgs(img, orientation, reserved=True)

    n2 = subimgs[0].shape[0]/2

    # Get the strips
    if orientation == TOP or orientation == BOTTOM:
        slice_ = (slice(0, n2), slice(0, 2*n2))
    else:
        slice_ = (slice(0, 2*n2), slice(0, n2))
    strips = tuple(subimg[slice_] for subimg in subimgs)

    # There are `rows` * `cols` layer-4 blocks in each element in strips
    if layer == 1:
        nrows, ncols = 8, 16
    elif layer == 2:
        nrows, ncols = 2, 4
    if orientation == LEFT or orientation == RIGHT:
        nrows, ncols = ncols, nrows

    strip_imgs = list()
    for strip in strips:
        for cols in np.split(strip, ncols, axis=1):
            for stripimg in np.split(cols, nrows, axis=0):
                strip_imgs.append(stripimg)

    return strip_imgs


def detect_color_and_orientation(img):
    """Detect the color and orientation of a reserved block.

    The return value is a list of (color, orientation) tuples that describe the
    detected color and orientation. If the block is ambiguous, the list contains
    more than one tuple."""
    assert img.shape[0] == img.shape[1]

    n2 = img.shape[0]/2
    top = img[:n2, :]
    bottom = img[n2:, :]
    left = img[:, :n2]
    right = img[:, n2:]

    # Find the slice that has most pixels of the same color.
    max_count = -1
    result = list()
    for orientation, slice_ in zip((TOP, LEFT, RIGHT, BOTTOM),
                                   (top, left, right, bottom)):
        nblack, nwhite = count_pixels(slice_)
        color = 1 if (nwhite > nblack) else 0
        count = max(nblack, nwhite)
        if count > max_count:
            result = [(color, orientation)]
            max_count = count
        elif count == max_count:
            result.append((color, orientation))

    return result


class Block(object):
    def __init__(self, layer=1, use_extra_strips=True):
        self.layer = layer          # Layer this block belongs to
        self.color = -1             # Color of the reserved block (1 bit)
        self.orientation = -1       # Orientation of reserved block (2 bits)
        self.bit_idxs = None        # Position of bits in payload
        self.children = list()      # List of 8 children, unless in last layer
        self.extra_children = list()    # Children in "extra strips"
        self.img = None             # A rendering of this block

        if self.layer < NLAYERS:
            self.children = [Block(layer=layer+1) for _ in xrange(8)]

        if use_extra_strips:
            if layer == 1:
                self.extra_children = [Block(layer=NLAYERS) for _ in xrange(1024)]
            elif layer == 2:
                self.extra_children = [Block(layer=NLAYERS) for _ in xrange(64)]

        if layer == 1:
            self._set_bit_idxs()

    def _set_bit_idxs(self):
        """Helper function to map blocks to bit positions in payload."""
        # Set bit positions for all blocks.
        bit_pos = 0
        queue = collections.deque()
        queue.append(self)
        while len(queue) > 0:
            block = queue.popleft()
            if block.layer < NLAYERS:
                block.bit_idxs = range(bit_pos, bit_pos+3)
                bit_pos += 3
            else:
                block.bit_idxs = (bit_pos, )
                bit_pos += 1
            queue.extend(block.children)
            queue.extend(block.extra_children)

    def encode(self, payload):
        """Encodes the payload in the block and its children."""
        if self.layer < NLAYERS:
            bit0, bit1, bit2 = [payload[i] for i in self.bit_idxs]
            self.color = bit0
            self.orientation = bit1 << 1 | bit2
        else:
            self.color = payload[self.bit_idxs[0]]

        for child in self.children + self.extra_children:
            child.encode(payload)

    def is_ambiguous(self):
        """Returns true if this block or any of its children is ambiguous."""
        if self.img is None:
            assert self.layer == 1
            self.render()

        if self.layer == NLAYERS:
            return False

        # Is one of our children ambiguous?
        for child in self.children:
            if child.is_ambiguous():
                return True

        _, orientations = zip(*detect_color_and_orientation(self.img))
        orientations = set(orientations)
        if len(orientations) > 1:
            return True

        return False

    def render(self, img=None, smallest_block_size=1):
        """Render the current block into `img`."""
        if img is None:
            img = np.empty((64*smallest_block_size, 64*smallest_block_size),
                           dtype=np.uint8)
            img[:, :] = -1

        assert img.shape[0] == img.shape[1], 'Image must be square.'
        self.img = img

        if len(self.children) == 0:
            # This block is in the deepest layer, so it only
            # has a color.
            img[:, :] = self.color
            return

        # Draw reserved block
        for subimg in get_subblock_imgs(img, self.orientation, reserved=True):
            subimg[:, :] = self.color

        # Draw children in reserved block
        self._render_extra_strips(img)

        # Recursively draw next layer
        subimgs = get_subblock_imgs(img, self.orientation)
        for block, subimg in zip(self.children, subimgs):
            block.render(subimg)

        return img

    def _render_extra_strips(self, img):
        """Render layer-4 blocks in extra strips."""
        if len(self.extra_children) == 0:
            return

        assert self.layer == 1 or self.layer == 2

        # Set the pixels in the extra strip
        colors = [rc.color for rc in self.extra_children]
        for eximg in get_extra_strip_imgs(img, self.orientation, self.layer):
            eximg[:, :] = colors.pop(0)

        # Should have consumed all children
        assert len(colors) == 0

    def decode(self, img, payload=None):
        """Decode bits from `img`."""
        assert img.shape[0] == img.shape[1]

        if payload is None:
            assert self.layer == 1
            payload = [-1 for _ in xrange(2267)]

        if len(self.children) == 0:
            # This is a block in the deepest layer.
            nblack, nwhite = count_pixels(img)
            self.color = 0 if nblack > nwhite else 1
            payload[self.bit_idxs[0]] = self.color
            return payload

        # Decode
        try:
            [(color, orientation)] = detect_color_and_orientation(img)
        except ValueError:
            color, orientation = detect_color_and_orientation(img)[0]
            print 'WARNING: Ambiguous block for bits', self.bit_idxs
            print img
            print '---'
        self.color, self.orientation = color, orientation

        idx0, idx1, idx2 = self.bit_idxs
        payload[idx0] = self.color
        payload[idx1] = (self.orientation & 2) >> 1
        payload[idx2] = self.orientation & 1

        # Decode children in extra strips
        self._decode_extra_strips(img, payload)

        # Recursively decode children
        subimgs = get_subblock_imgs(img, self.orientation)
        for block, subimg in zip(self.children, subimgs):
            block.decode(subimg, payload=payload)

        return payload

    def _decode_extra_strips(self, img, payload):
        """Decode layer-4 bits from extra strips."""
        if len(self.extra_children) == 0:
            return

        assert self.layer == 1 or self.layer == 2

        eximgs = get_extra_strip_imgs(img, self.orientation, self.layer)
        assert len(eximgs) == len(self.extra_children)
        for eximg, subblock in zip(eximgs, self.extra_children):
            subblock.decode(eximg, payload=payload)


def random_payload(seed=2610, n=2267):
    """Generate a pseudorandom payload from seed."""
    rand = random.Random(seed)
    return [rand.randint(0, 1) for _ in xrange(n)]


def test_detect_color_and_orientation():
    img = np.zeros((16, 16))
    colors, orientations = zip(*detect_color_and_orientation(img))
    orientations = set(orientations)
    assert orientations == set((TOP, LEFT, BOTTOM, RIGHT)) and \
        len(set(colors)) == 1 and colors[0] == 0

    img[:8, :] = 1
    colors, orientations = zip(*detect_color_and_orientation(img))
    orientations = set(orientations)
    assert orientations == set((TOP, BOTTOM)) and colors == (1, 0)

    img[:, :8] = 1
    colors, orientations = zip(*detect_color_and_orientation(img))
    orientations = set(orientations)
    assert orientations == set((TOP, LEFT)) and colors == (1, 1)

    img = np.fliplr(img)
    colors, orientations = zip(*detect_color_and_orientation(img))
    orientations = set(orientations)
    assert orientations == set((TOP, RIGHT)) and colors == (1, 1)

    img[-1, -1] = 0
    colors, orientations = zip(*detect_color_and_orientation(img))
    orientations = set(orientations)
    assert orientations == set((TOP, )) and colors == (1, )


def test_encode_decode():
    payload = random_payload(9320)
    code = Block()
    code.encode(payload)
    img = code.render()
    assert payload == Block().decode(img)


def main():
    test_detect_color_and_orientation()
    test_encode_decode()


if __name__ == '__main__':
    main()
