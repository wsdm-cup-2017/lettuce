# -*- coding: utf-8 -*-

import mwparserfromhell


class Paragraph(object):
    __slots__ = ('text', 'wiki_links', 'abstract')

    def __init__(self, text, wiki_links, abstract):
        self.text = text
        self.wiki_links = wiki_links
        self.abstract = abstract

    def __repr__(self):
        return '<Paragraph %s>' % (self.text[:20].encode('utf-8') + '...')

    def __reduce__(self):
        return (self.__class__, (self.text, self.wiki_links, self.abstract))


class WikiLink(object):
    __slots__ = ('title', 'text', 'link_type')

    def __init__(self, title, text, link_type='out'):
        self.title = title
        self.text = text
        self.link_type = link_type

    def __repr__(self):
        return '<WikiLink %s>' % self.title.encode('utf-8')

    def __reduce__(self):
        return (self.__class__, (self.title, self.text, self.link_type))


class WikiExtractor(object):
    def __init__(self, entity_db=None):
        self._entity_db = entity_db

    def extract_paragraphs(self, page):
        paragraphs = []
        cur_text = []
        cur_links = []

        if page.is_redirect:
            return []

        abstract = True

        for node in self._parse_page(page).nodes:
            if isinstance(node, mwparserfromhell.nodes.Text):
                for (n, paragraph) in enumerate(unicode(node).split('\n')):
                    if n == 0:
                        cur_text.append(paragraph)
                    else:
                        paragraphs.append(Paragraph(
                            u' '.join(cur_text), cur_links, abstract)
                        )
                        cur_text = [paragraph]
                        cur_links = []

            elif isinstance(node, mwparserfromhell.nodes.Heading):
                abstract = False

            elif isinstance(node, mwparserfromhell.nodes.Wikilink):
                title = node.title.strip_code()
                if not title:
                    continue

                title = self._normalize_title(title)
                if self._entity_db is not None:
                    title = self._entity_db.resolve_redirect(title)

                if node.text:
                    text = node.text.strip_code()
                else:
                    text = node.title.strip_code()

                cur_text.append(text)
                cur_links.append(WikiLink(title, text))

            elif isinstance(node, mwparserfromhell.nodes.Tag):
                if node.tag not in ('b', 'i'):
                    continue
                if not node.contents:
                    continue

                text = node.contents.strip_code()
                cur_text.append(text)

        return paragraphs

    def _parse_page(self, page):
        try:
            return mwparserfromhell.parse(page.wiki_text)
        except Exception:
            return mwparserfromhell.parse('')

    @staticmethod
    def _normalize_title(title):
        return (title[0].upper() + title[1:]).replace('_', ' ')
