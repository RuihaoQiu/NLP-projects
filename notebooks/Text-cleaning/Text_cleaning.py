from unidecode import unidecode
import re

class TextCleaner(object):
    """The module includes cleaning functions for different scenario.
    """
    def __init__(self):
        self.CURRENCIES = {'$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY'}
        self.REDUNDANT_TERMS = {"Apply Save Share", "Apply Now", "About us"}

        ## Regular expressions
        self.newline_regex = re.compile(r"[\n\r]+")
        self.double_newline_regex = re.compile(r"\n[\s]*\n")
        self.space_regex = re.compile(r"[\s]+")
        self.space_only_regex = re.compile(r" +")

        self.url_regex = re.compile(r"[\(\<]http.+?[\>\)]")
        self.email_regex = re.compile(r"[\w.-]+@[\w.-]+")

        self.bad_char_regex = re.compile(r"[;:\!\?\=\"\'\*\•\’\‘\%\/\]\[\)\(\>\<\|\!\_\\]+")
        self.some_char_regex = re.compile(r"[;:\!\?\=\"\'\’\‘\%\/\]\[\)\(\>\<\|\!\_\\]+") ## exclude the list indicators like [-*•] ...
        self.normal_char_regex = re.compile(r"[,.]+")
        self.hypen_regex = re.compile(r"(\s\-+\s*|\s*\-+\s)") # hypen between space
        self.and_regex = re.compile(r"(\s\&+\s*|\s*\&+\s)")

        self.single_char_regex = re.compile(r"(?<=[,.\s])[^RIC](?=[,.\s])", re.IGNORECASE)
        self.long_char_regex = re.compile(r"[^\s]{25,}")
        self.hash_regex = re.compile(r"(?<!C)#", re.IGNORECASE)
        self.plus_regex = re.compile(r"(?<!C)(?<!\+)\++", re.IGNORECASE)
        self.separated_number_regex = re.compile(r"\b\d+[,.\s\-]+\d+\b")
        self.number_regex = re.compile(r"\b\d+\b")
        self.contain_number_regex = re.compile(r"[\d]+")

        self.currency_regex = re.compile('({})+'.format('|'.join(re.escape(c) for c in self.CURRENCIES.keys())))
        self.redundant_terms_regex = re.compile('({})+'.format('|'.join(re.escape(c) for c in self.REDUNDANT_TERMS)), re.IGNORECASE)

        self.abbr_regex = re.compile(r"(e\.g\.|i\.e\.|etc[\.]*)", re.IGNORECASE)
        self.javascript_plugin_regex = re.compile(r"javascript:")
        self.find_all_jobs_regex = re.compile(r"\(.+?\"Find all jobs matching.+?\)")

    def paragraph_segment(self, text):
        return self.double_newline_regex.split(text)

    def remove_newlines(self, text):
        return self.newline_regex.sub(" ", text)

    def remove_space(self, text):
        return self.space_regex.sub(" ", text.strip())

    def remove_space_only(self, text):
        return self.space_only_regex.sub(" ", text).strip()

    def remove_url(self, text):
        return self.url_regex.sub(" ", text)

    def remove_email(self, text):
        return self.email_regex.sub(" ", text)

    def remove_bad_char(self, text):
        return self.bad_char_regex.sub(" ", text)

    def remove_some_char(self, text):
        return self.some_char_regex.sub(" ", text)

    def remove_hash(self, text):
        return self.hash_regex.sub(" ", text)

    def remove_plus(self, text):
        return self.plus_regex.sub(" ", text)

    def remove_and(self, text):
        return self.and_regex.sub(" ", text)

    def remove_hypen(self, text):
        return self.hypen_regex.sub(" ", text)

    def remove_normal_char(self, text):
        return self.normal_char_regex.sub(" ", text)

    def remove_single_char(self, text):
        return self.single_char_regex.sub(" ", text)

    def remove_abbr(self, text):
        return self.abbr_regex.sub("", text)

    def remove_numbers(self, text):
        return self.number_regex.sub("", text)

    def remove_separated_numbers(self, text):
        return self.separated_number_regex.sub("", text)

    def remove_currency(self, text):
        return self.currency_regex.sub("", text)

    def remove_redundant_terms(self, text):
        return self.redundant_terms_regex.sub("", text)

    def remove_long_char(self, text):
        return self.long_char_regex.sub("", text)

    def remove_javascript_plugin(self, text):
        if self.javascript_plugin_regex.search(text):
            return ""
        else:
            return text

    def remove_find_all_jobs(self, text):
        return self.find_all_jobs_regex.sub("", text)

    def remove_words_with_number(self, text):
        text_list = text.split()
        text_out = [word for word in text_list if not self.contain_number_regex.search(word)]
        return " ".join(text_out)
