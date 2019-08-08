import gzip
import json

from dioptre.data.generator import DataGenerator


class JSONLine(DataGenerator):
    """Reads and render examples from jsonline file with dictionaries containing `text` key.

    Arguments:
        filename: path to a raw or gzipped `.jsonline` file.
    """

    def __init__(self, alphabet, filename, **kwargs):
        super().__init__(alphabet, **kwargs)
        self.filename = filename

    def iterate_text(self):
        if self.filename.endswith('.gz'):
            fopen = gzip.GzipFile
        else:
            fopen = open

        with fopen(self.filename) as fp:
            for line in fp:
                yield json.loads(line)['text']
