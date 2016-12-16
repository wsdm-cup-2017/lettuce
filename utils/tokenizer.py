# -*- coding: utf-8 -*-

import re


class DefaultTokenizer(object):
    def __init__(self, rule=ur'[\w\d]+'):
        self._rule = re.compile(rule, re.UNICODE)

    def tokenize(self, text):
        return [text[o.start():o.end()] for o in self._rule.finditer(text)]

    def span_tokenize(self, text):
        return [(o.start(), o.end()) for o in self._rule.finditer(text)]
