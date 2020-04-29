import argparse
import collections
import re


Section = collections.namedtuple('Section', 'index name attributes')


parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int)
parser.add_argument('--classes', type=int)
parser.add_argument('--no-color-adjustments', action='store_true')
parser.add_argument('--size', nargs=2, type=int)
parser.add_argument('--subdivisions', type=int)
parser.add_argument('baseconfig')
args = parser.parse_args()


# Parse the base configuration
config = []
with open(args.baseconfig) as f:
    for line in f:
        m = re.match(r'\[(.*?)\]', line)
        if m:
            config.append(Section(
                index=len(config),
                name=m.group(1),
                attributes={},
            ))
        
        m = re.match(r'\s*(\w+)\s*=\s*(.*?)\s*$', line)
        if m:
            key, value = m.group(1), m.group(2)
            config[-1].attributes[key] = value


def getsection(name):
    return next(getsections(name))

def getsections(name):
    return filter(lambda x: x.name == name, config)


# Make adjustments to configuration
net = getsection('net')

if args.batch is not None:
    net.attributes['batch'] = args.batch

if args.classes is not None:
    net.attributes['max_batches'] = max_batches = \
        max(4000, args.classes * 2000)
    net.attributes['steps'] = \
        f'{int(0.8*max_batches)},{int(0.9*max_batches)}'
    
    for section in getsections('yolo'):
        section.attributes['classes'] = args.classes
        config[section.index-1].attributes['filters'] = (args.classes + 5) * 3
    
    for section in getsections('gaussian_yolo'):
        section.attributes['classes'] = args.classes
        config[section.index-1].attributes['filters'] = (args.classes + 9) * 3

if args.no_color_adjustments == True:
    for attr in ('hue', 'saturation', 'exposure'):
        try:
            del net.attributes[attr]
        except KeyError:
            pass

if args.size is not None:
    assert args.size[0] % 32 == 0
    assert args.size[1] % 32 == 0
    net.attributes['width'] = args.size[0]
    net.attributes['height'] = args.size[1]

if args.subdivisions is not None:
    net.attributes['subdivisions'] = args.subdivisions


# Serialize out the new configuration
for section in config:
    if section.index > 0:
        print()
    print(f'[{section.name}]')
    for key, value in section.attributes.items():
        print(f'{key} = {value}')
