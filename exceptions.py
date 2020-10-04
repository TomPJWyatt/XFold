class UnknownTifMeta(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'The tiff file you provided did not contain'\
                'metadata of a format that we can handle.'
