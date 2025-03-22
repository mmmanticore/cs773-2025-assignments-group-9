#!/usr/bin/env python

# smoothing.py - Smoothing filter operations on pixel arrays in 2D
#
# Copyright (C) 2020 Martin Urschler <martin.urschler@auckland.ac.nz>
#
# Original concept by Martin Urschler.
#
# LICENCE (MIT)
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import imageProcessing.utilities as IPUtils
import imageProcessing.convolve2D as IPConv2D



# def computeGaussianAveraging3x3(pixel_array, image_width, image_height):

#     # sigma is 3 pixels
#     smoothing_3tap = [0.27901, 0.44198, 0.27901]

#     averaged = IPConv2D.computeSeparableConvolution2DOddNTapBorderZero(pixel_array, image_width, image_height, smoothing_3tap)

#     return averaged

def computeGaussianAveraging(pixel_array, image_width, image_height, kernel_size=3):
    """
    通用高斯平滑函数，支持 kernel_size=3, 5, 7 自动选择卷积核
    """
    if kernel_size == 3:
        kernel = [0.27901, 0.44198, 0.27901]
    elif kernel_size == 5:
        kernel = [0.0625, 0.25, 0.375, 0.25, 0.0625]  # σ≈1.0
    elif kernel_size == 7:
        kernel = [0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]  # σ≈1.5
    elif kernel_size == 9:
            kernel = [0.00098, 0.00876, 0.02698, 0.06476, 0.09678,
                    0.06476, 0.02698, 0.00876, 0.00098]  # σ≈2.0
    else:
        raise ValueError("Only support kernel_size = 3, 5, 7, or 9")


    averaged = IPConv2D.computeSeparableConvolution2DOddNTapBorderZero(
        pixel_array, image_width, image_height, kernel
    )
    return averaged




