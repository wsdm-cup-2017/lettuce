# -*- coding: utf-8 -*-

import click
import re
from collections import defaultdict
from contextlib import closing
from shelve import DbfilenameShelf
from multiprocessing.pool import Pool
from wiki_dump_reader import WikiDumpReader
from wiki_extractor import WikiExtractor, WikiLink

MULTI_SPACE_RE = re.compile(ur'\s+')

_extractor = None


class PageDB(DbfilenameShelf):
    @staticmethod
    def build(dataset, entity_db, dump_file, out_file, category, pool_size,
              chunk_size):
        global _extractor

        dump_reader = WikiDumpReader(dump_file)
        _extractor = WikiExtractor(entity_db)

        if category == 'pro':
            kb_data = dataset.profession_kb
            click.echo('Category: Profession')

        elif category == 'nat':
            kb_data = dataset.nationality_kb
            click.echo('Category: Nationality')

        else:
            raise RuntimeError('Invalid category')

        target_titles = frozenset([entity_db.resolve_redirect(title)
                                   for (title, _) in kb_data])
        paragraph_buf = {}
        link_buf = defaultdict(list)

        with closing(Pool(pool_size)) as pool:
            for (title, paragraphs) in pool.imap_unordered(
                _process_page, dump_reader, chunksize=chunk_size
            ):
                if title in target_titles:
                    paragraph_buf[title] = paragraphs

                for paragraph in paragraphs:
                    for link in paragraph.wiki_links:
                        if link.title in target_titles:
                            link_buf[link.title].append(WikiLink(title, link.text, 'in'))

        with closing(PageDB(out_file, protocol=-1)) as db:
            with click.progressbar(paragraph_buf.iteritems(), length=len(paragraph_buf)) as bar:
                for (title, paragraphs) in bar:
                    db[title.encode('utf-8')] = dict(
                        paragraphs=paragraphs, in_links=link_buf[title]
                    )

    def get_paragraphs(self, title, abstract_only=False):
        paragraphs = self[title.encode('utf-8')]['paragraphs']

        if abstract_only:
            return [a for a in paragraphs if a.abstract]
        else:
            return paragraphs

    def get_text(self, title, abstract_only=False):
        text = u' '.join(p.text for p in self.get_paragraphs(title, abstract_only))
        return MULTI_SPACE_RE.sub(u' ', text)

    def get_links(self, title, in_links=True, out_links=True, abstract_only=False):
        ret = []
        if out_links:
            ret += [l for p in self.get_paragraphs(title, abstract_only) for l in p.wiki_links]

        if in_links:
            ret += self[title.encode('utf-8')]['in_links']

        return ret


def _process_page(page):
    return (page.title, _extractor.extract_paragraphs(page))
