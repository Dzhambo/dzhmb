import socket
import time
from collections import defaultdict

class ClientError(Exception):
    pass

class Client:
    def __init__(self, HOST, PORT, timeout=None):
        self.HOST = HOST
        self.PORT = PORT
        self.timeout = timeout

    def put(self, name_metric, value_metric, timestamp=None):
        if not timestamp:
            timestamp = int(time.time())
        message = ' '.join(['put', str(name_metric), str(value_metric), str(timestamp)+'\n'])
        with socket.create_connection((self.HOST, self.PORT), self.timeout) as sock:
            sock.send(message.encode())
            data = sock.recv(1024).decode()
        if data == 'error\nwrong command\n\n':
            raise ClientError()

    def get(self,name_metric):
        with socket.create_connection((self.HOST, self.PORT), self.timeout) as sock:
            key = 'get {}\n'.format(name_metric)
            sock.send(key.encode())
            data = sock.recv(1024).decode()
        if data == 'ok\n\n':
            return {}
        if data == 'error\nwrong command\n\n':
            raise ClientError()
        if data.rstrip('\n\n').split('\n')[0]!='ok':
            raise ClientError()


        metric_items = data.lstrip('ok\n').rstrip('\n\n')
        metric_items = [item.split() for item in metric_items.split('\n')]
        items_dict = defaultdict(list)
        try:
            for key, name_metric, timestamp in metric_items:
                items_dict[key].append((int(timestamp), float(name_metric)))
        except Exception:
            raise ClientError()

        for key in items_dict:
            items_dict[key].sort(key=lambda x: x[0])

        return items_dict

