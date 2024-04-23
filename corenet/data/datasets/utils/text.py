#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import re
import urllib

import ftfy


def caption_preprocessing(caption: str) -> str:
    """Removes the unwanted tokens (e.g., HTML tokens, next line, unwanted spaces) from
    the text."""
    # captions may contain HTML tokens. Remove them
    html_re = re.compile("<.*?>")
    caption = urllib.parse.unquote(str(caption))
    caption = caption.replace("+", " ")
    caption = re.sub(html_re, "", str(caption))
    # remove the next line
    caption = caption.strip("\n")
    # remove unwanted spaces
    caption = re.sub(" +", " ", caption)

    caption = ftfy.fix_text(caption)
    return caption.strip().lower()
