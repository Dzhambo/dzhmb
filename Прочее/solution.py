class FileReader:

    def __init__(self, filepath):
        self.path = filepath

    def read(self):
        try:
            with open(self.path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return ''


