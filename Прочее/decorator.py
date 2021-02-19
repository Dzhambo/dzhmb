import functools, json


def to_json(func):
    @functools.wraps(func)
    def wrapped(*args,**kwargs):
        return json.dumps(func(*args,**kwargs))
    return wrapped
