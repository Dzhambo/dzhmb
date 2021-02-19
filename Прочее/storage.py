import argparse, os, json, tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--key")
parser.add_argument("--val")
args = parser.parse_args()

storage_path = os.path.join(tempfile.gettempdir(), 'storage.data')

if os.path.isfile(storage_path):
    if args.val:
        with open(storage_path, "r") as f:
            m = json.load(f)
            if args.key in m:
                m[args.key] = m[args.key] + [args.val]
            else:
                m.update({args.key: [args.val]})
        with open(storage_path, "w") as f:
            json.dump(m, f)
    elif args.key:
        try:
            with open(storage_path, "r") as f:
                m = json.load(f)
                if m[args.key] == None:
                    print(None)
                if len(m[args.key]) > 1:
                    print(', '.join(m.get(args.key)))
                else:
                    print(m.get(args.key)[0])
        except:
            print(None)
else:
    d = {}
    with open(str(storage_path), "w") as f:
        if args.val:
            d = {args.key: [args.val]}
            json.dump(d, f)
        else:
            print(None)