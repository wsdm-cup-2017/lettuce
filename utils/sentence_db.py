# -*- coding: utf-8 -*-

import click
import re
from contextlib import closing
from shelve import DbfilenameShelf

LINK_RE = re.compile(ur'\[(.+?)\|(.*?)\]')
WIKI_SENTENCES_LINE_LEN = 33159353


class Sentence(object):
    __slots__ = ('text', 'wiki_links')

    def __init__(self, text, wiki_links):
        self.text = text
        self.wiki_links = wiki_links

    def __repr__(self):
        return '<Sentence %s>' % (self.text[:20].encode('utf-8') + '...')

    def __reduce__(self):
        return (self.__class__, (self.text, self.wiki_links))


class WikiLink(object):
    __slots__ = ('title', 'span')

    def __init__(self, title, span):
        self.title = title
        self.span = span

    def __repr__(self):
        return '<WikiLink %s>' % self.title.encode('utf-8')

    def __reduce__(self):
        return (self.__class__, (self.title, self.span))


class SentenceDB(DbfilenameShelf):
    @classmethod
    def build(cls, wiki_sentences, entity_db, out_file):
        with closing(SentenceDB(out_file, protocol=-1)) as db:
            with click.progressbar(wiki_sentences, length=WIKI_SENTENCES_LINE_LEN) as bar:
                for (n, line) in enumerate(bar):
                    sent = cls._parse_sentence(line.rstrip().decode('utf-8'), entity_db)
                    db[str(n)] = sent

    @staticmethod
    def _parse_sentence(text, entity_db):
        wiki_links = []

        while True:
            match_obj = LINK_RE.search(text)
            if not match_obj:
                break

            title = match_obj.group(1)
            title = title.replace(u'_', u' ')
            title = entity_db.resolve_redirect(title)

            link_text = match_obj.group(2)

            start = match_obj.start()
            end = match_obj.end()
            text = text[:start] + link_text + text[end:]

            wiki_links.append(WikiLink(title, (start, start + len(link_text))))

        return Sentence(text, wiki_links)
