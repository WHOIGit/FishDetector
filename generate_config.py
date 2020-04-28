import argparse
import itertools
import re


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--set', nargs=2, action='append')
parser.add_argument('-u', '--unset', action='append')
parser.add_argument('baseconfig')
args = parser.parse_args()
print(args)


# Parse the base configuration
config = []
with open(args.baseconfig) as f:
    for line in f:
        m = re.match(r'\[(.*?)\]', line)
        if m:
            config.append((m.group(1), {}))
        
        m = re.match(r'\s*(\w+)\s*=\s*(.*?)\s*$', line)
        if m:
            key, value = m.group(1), m.group(2)
            config[-1][1][key] = value


def parse_key(key):
    m = re.match(r'(\w+)(?:\[(-?\d*)(?::(-?\d*))?(?::(-?\d*))?\])?\.(\w+)', key)
    group, sliceidx, key = m.groups()[0], m.groups()[1:-1], m.groups()[-1]
    if all(g is None for g in sliceidx):
        slc = slice(0, 1)
    else:
        slc = slice(*(None if x == '' or x is None else int(x)
                      for x in sliceidx))
    return group, slc, key


# Handle sets
for key, value in args.set:
    group, slc, key = parse_key(key)
    for g in list(filter(lambda x: x[0] == group, config))[slc]:
        g[1][key] = value

# Handle unsets
for key in args.unset:
    group, slc, key = parse_key(key)
    for g in list(filter(lambda x: x[0] == group, config))[slc]:
        del g[1][key]


# Serialize out the new configuration
for i, (heading, key_values) in enumerate(config):
    if i > 0:
        print()
    print(f'[{heading}]')
    for key, value in key_values.items():
        print(f'{key} = {value}')
