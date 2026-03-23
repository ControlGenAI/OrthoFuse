import numpy as np

import cv2

import skimage.io as io

import PIL
from PIL import Image

import matplotlib.pyplot as plt

object_prompts_list = [
    'a {0} {1} in the jungle',
    'a {0} {1} in the snow',
    'a {0} {1} on the beach',
    'a {0} {1} on a cobblestone street',
    'a {0} {1} on top of pink fabric',
    'a {0} {1} on top of a wooden floor',
    'a {0} {1} with a city in the background',
    'a {0} {1} with a mountain in the background',
    'a {0} {1} with a blue house in the background',
    'a {0} {1} on top of a purple rug in a forest',
    'a {0} {1} with a wheat field in the background',
    'a {0} {1} with a tree and autumn leaves in the background',
    'a {0} {1} with the Eiffel Tower in the background',
    'a {0} {1} floating on top of water',
    'a {0} {1} floating in an ocean of milk',
    'a {0} {1} on top of green grass with sunflowers around it',
    'a {0} {1} on top of a mirror',
    'a {0} {1} on top of the sidewalk in a crowded street',
    'a {0} {1} on top of a dirt road',
    'a {0} {1} on top of a white rug',
    'a red {0} {1}',
    'a purple {0} {1}',
    'a shiny {0} {1}',
    'a wet {0} {1}',
    'a cube shaped {0} {1}',
]

live_prompts_list = [
    'a {0} {1} in the jungle',
    'a {0} {1} in the snow',
    'a {0} {1} on the beach',
    'a {0} {1} on a cobblestone street',
    'a {0} {1} on top of pink fabric',
    'a {0} {1} on top of a wooden floor',
    'a {0} {1} with a city in the background',
    'a {0} {1} with a mountain in the background',
    'a {0} {1} with a blue house in the background',
    'a {0} {1} on top of a purple rug in a forest',
    'a {0} {1} wearing a red hat',
    'a {0} {1} wearing a santa hat',
    'a {0} {1} wearing a rainbow scarf',
    'a {0} {1} wearing a black top hat and a monocle',
    'a {0} {1} in a chef outfit',
    'a {0} {1} in a firefighter outfit',
    'a {0} {1} in a police outfit',
    'a {0} {1} wearing pink glasses',
    'a {0} {1} wearing a yellow shirt',
    'a {0} {1} in a purple wizard outfit',
    'a red {0} {1}',
    'a purple {0} {1}',
    'a shiny {0} {1}',
    'a wet {0} {1}',
    'a cube shaped {0} {1}',
]

eval_prompts_list = [
    'a cube shaped {0} {1}',
    'a purple {0} {1}',
    'a red {0} {1}',
    'a shiny {0} {1}',
    'a wet {0} {1}',
    'a {0} {1} floating in an ocean of milk',
    'a {0} {1} floating on top of water',
    'a {0} {1} in a chef outfit',
    'a {0} {1} in a firefighter outfit',
    'a {0} {1} in a police outfit',
    'a {0} {1} in a purple wizard outfit',
    'a {0} {1} in the jungle',
    'a {0} {1} in the snow',
    'a {0} {1} on a cobblestone street',
    'a {0} {1} on the beach',
    'a {0} {1} on top of a dirt road',
    'a {0} {1} on top of a mirror',
    'a {0} {1} on top of a purple rug in a forest',
    'a {0} {1} on top of a white rug',
    'a {0} {1} on top of a wooden floor',
    'a {0} {1} on top of green grass with sunflowers around it',
    'a {0} {1} on top of pink fabric',
    'a {0} {1} on top of the sidewalk in a crowded street',
    'a {0} {1} wearing a black top hat and a monocle',
    'a {0} {1} wearing a rainbow scarf',
    'a {0} {1} wearing a red hat',
    'a {0} {1} wearing a santa hat',
    'a {0} {1} wearing a yellow shirt',
    'a {0} {1} wearing pink glasses',
    'a {0} {1} with a blue house in the background',
    'a {0} {1} with a city in the background',
    'a {0} {1} with a mountain in the background',
    'a {0} {1} with a tree and autumn leaves in the background',
    'a {0} {1} with a wheat field in the background',
    'a {0} {1} with the Eiffel Tower in the background'
]

training_prompts_list = [
    'a pyramid shaped {0} {1}',
    'a yellow {0} {1}',
    'a green {0} {1}',
    'a dirty {0} {1}',
    'a bright {0} {1}',
    'a {0} {1} floating in an lake of juice',
    'a {0} {1} floating on top of grass',
    'a {0} {1} in a doctor outfit',
    'a {0} {1} in a scientist outfit',
    'a {0} {1} in a postman outfit',
    'a {0} {1} in a blue farmer outfit',
    'a {0} {1} in the savanna',
    'a {0} {1} in the sky',
    'a {0} {1} in a dark alley',
    'a {0} {1} in a lush forest',
    'a {0} {1} on top of a black and white checkered tile floor',
    'a {0} {1} on top of a stack of books in a cozy library',
    'a {0} {1} on top of a red and white checkered picnic blanket',
    'a {0} {1} on top of a silver metal bench',
    'a {0} {1} on top of a blue velvet pillow',
    'a {0} {1} on top of a black leather couch',
    'a {0} {1} on top of a concrete floor with graffiti',
    'a {0} {1} on top of a snowy mountain peak',
    'a {0} {1} wearing a green bow tie',
    'a {0} {1} carrying a brown leather messenger bag',
    'a {0} {1} wearing a blue bandana and aviator sunglasses',
    'a {0} {1} wearing a white lab coat',
    'a {0} {1} with a pink umbrella',
    'a {0} {1} wearing a red hoodie',
    'a {0} {1} with a sunset and palm trees in the background',
    'a {0} {1} with a city skyline at night in the background',
    'a {0} {1} with a castle in the background',
    'a {0} {1} with a forest in the background',
    'a {0} {1} with the Sydney Opera House in the background',
    'a {0} {1} with a rocky coastline in the background',
]

classes_data = {
    # 'person':                 ('sks', '<person>',         'person'        ),    # noqa
    # 'me':                     ('sks', '<me-person>',      'person'        ),    # noqa
    'backpack':               ('sks', '<backpack>',       'backpack'      ),    # noqa
    # 'backpack1':              ('sks', '<backpack>',       'backpack'      ),    # noqa
    'backpack_dog':           ('sks', '<backpack>',       'backpack'      ),    # noqa
    'bear_plushie':           ('sks', '<stuffed animal>', 'stuffed animal'),    # noqa
    'berry_bowl':             ('sks', '<bowl>',           'bowl'          ),    # noqa
    'can':                    ('sks', '<can>',            'can'           ),    # noqa
    'candle':                 ('sks', '<candle>',         'candle'        ),    # noqa
    'cat':                    ('sks', '<cat>',            'cat'           ),    # noqa
    'cat2':                   ('sks', '<cat>',            'cat'           ),    # noqa
    'clock':                  ('sks', '<clock>',          'clock'         ),    # noqa
    'colorful_sneaker':       ('sks', '<sneaker>',        'sneaker'       ),    # noqa
    'dog':                    ('sks', '<dog>',            'dog'           ),    # noqa
    'dog2':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog3':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog5':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog6':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog7':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog8':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'duck_toy':               ('sks', '<toy>',            'toy'           ),    # noqa
    'fancy_boot':             ('sks', '<boot>',           'boot'          ),    # noqa
    'grey_sloth_plushie':     ('sks', '<stuffed animal>', 'stuffed animal'),    # noqa
    'monster_toy':            ('sks', '<toy>',            'toy'           ),    # noqa
    'pink_sunglasses':        ('sks', '<glasses>',        'glasses'       ),    # noqa
    'poop_emoji':             ('sks', '<toy>',            'toy'           ),    # noqa
    'rc_car':                 ('sks', '<toy>',            'toy'           ),    # noqa
    'red_cartoon':            ('sks', '<cartoon>',        'cartoon'       ),    # noqa
    'robot_toy':              ('sks', '<toy>',            'toy'           ),    # noqa
    'shiny_sneaker':          ('sks', '<sneaker>',        'sneaker'       ),    # noqa
    'teapot':                 ('sks', '<teapot>',         'teapot'        ),    # noqa
    'vase':                   ('sks', '<vase>',           'vase'          ),    # noqa
    'wolf_plushie':           ('sks', '<stuffed animal>', 'stuffed animal'),    # noqa
}

live_object_data = {
    'person':                 'live',    # noqa
    'backpack':               'object',    # noqa
    'stuffed animal':         'object',    # noqa
    'bowl':                   'object',    # noqa
    'can':                    'object',    # noqa
    'candle':                 'object',    # noqa
    'cat':                    'live',    # noqa
    'clock':                  'object',    # noqa
    'sneaker':                'object',    # noqa
    'dog':                    'live',    # noqa
    'toy':                    'object',    # noqa
    'boot':                   'object',    # noqao
    'cartoon':                'object',    # noqa
    'teapot':                 'object',    # noqa
    'vase':                   'object',    # noqa
    'glasses':                'object',
}

_LOAD_IMAGE_BACKENDS = {
    'PIL': lambda path: np.asarray(PIL.Image.open(path).convert('RGB')),
    'plt': lambda path: plt.imread(path),
    'skimage': lambda path: io.imread(path),
    'opencv': lambda path: cv2.imread(path),
    None: None,
}
_LOAD_IMAGE_BACKEND = _LOAD_IMAGE_BACKENDS['PIL']
