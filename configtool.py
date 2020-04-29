import argparse
import json
import re


parser = argparse.ArgumentParser()
parser.add_argument('--to-json', action='store_true')
parser.add_argument('--to-cfg', dest='to_json', action='store_false')
parser.add_argument('baseconfig')
args = parser.parse_args()

if args.baseconfig == '-':
    args.baseconfig = '/dev/stdin'


if args.to_json:
    # Parse the base configuration
    config = []
    with open(args.baseconfig) as f:
        for line in f:
            m = re.match(r'\[(.*?)\]', line)
            if m:
                config.append({
                    'section': m.group(1),
                    'attributes': {},
                })
            
            m = re.match(r'\s*(\w+)\s*=\s*(.*?)\s*$', line)
            if m:
                key, value = m.group(1), m.group(2)
                config[-1]['attributes'][key] = value

    # Add previous and next section hints
    for i, group in enumerate(config):
        group['prev'] = config[i]['section'] if i > 0 else None
        group['next'] = config[i+1]['section'] if i < len(config) - 1 else None

    print(json.dumps(config))

else:
    with open(args.baseconfig) as f:
        config = json.load(f)
    
    # Serialize out the new configuration
    for i, obj in enumerate(config):
        if i > 0:
            print()
        print(f'[{obj["section"]}]')
        for key, value in obj['attributes'].items():
            print(f'{key} = {value}')
