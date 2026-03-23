from .configs import eval_prompts_list as joined_set


appearance_set = [
    'a chocolate-colored {0}',
    'a cube shaped {0}',
    'a snowy {0}',
    'a spotted {0}',
    'a tiny {0}',
    'a wet {0}',
    'a dirty {0}',
    'a rainbow-colored {0}',
    'a striped {0}',
    'a frosted {0}',
    'a silver {0}',
    'a sandy {0}',
    'a {0} made of wooden',
    'a {0} made of paper',
    'a brightly lit {0}',
    'a plump {0}',
    'a trendy {0}',
    'a stylish {0}',
    'a pyramid shaped {0}',
    'an ugly {0}'
]
outfit_set = [
    'a {0} in a purple wizard outfit',
    'a {0} in the tuxedo',
    'a {0} in a doctor outfit',
    'a {0} wearing a red hat',
    'a {0} wearing pink glasses',
    'a {0} wearing a black top hat and a monocle',
    'a {0} in a firefighter outfit',
    'a {0} wearing a green bow tie',
    'a {0} with a pink umbrella',
    'a {0} wearing a red hoodie',
    'a {0} in a police outfit',
    'a {0} wearing a santa hat',
    'a {0} wearing a yellow shirt',
    'a {0} wearing a rainbow scarf',
    'a {0} as an astronaut',
    'a {0} with high-tech combat suit',
    'a {0} with a tiara',
    'a {0} in a knight suit',
    'a {0} in a helmet',
    'a {0} in a superhero suit'
]

background_set = [
    'a {0} with the Eiffel Tower in the background',
    'a {0} with a mountain in the background',
    'a {0} in the space',
    'a {0} on top of green grass with sunflowers around it',
    'a {0} with a sunset and palm trees in the background',
    'a {0} in the jungle',
    'a {0} in the snow',
    'a {0} on the beach',
    'a {0} on top of a wooden floor',
    'a {0} floating on top of water',
    'a {0} on top of the sidewalk in a crowded street',
    'a {0} with a tree and autumn leaves in the background',
    'a {0} in the savanna',
    'a {0} on top of a red and white checkered picnic blanket',
    'a {0} on top of a concrete floor with graffiti',
    'a {0} on top of a snowy mountain peak',
    'a {0} with a castle in the background',
    'a {0} with the Sydney Opera House in the background',
    'a {0} in a cave',
    'a {0} in the Acropolis'
]

style_set = [
    'a {0} in watercolor painting style',
    'a {0} in anime style',
    'a {0} in Pixar animation style',
    'a {0} in sketch painting style',
    'a {0} in pop-art style',
    'a {0} in line drawing style',
    'a {0} in abstract cartoon style',
    'a {0} in 3d rendering style',
    'a {0} in a Funko Pop',
    'a {0} in kid crayon drawing style',
    'a {0} in a Pixel Art style',
    'a {0} in Cubism style',
    'a {0} in the style of Wassily Kandinsky',
    'a {0} in the style of Vincent van Gogh',
    'a {0} in impressionism style',
    'a {0} in graffiti style',
    'a {0} in cyberpunk style',
    'a {0} in Disney animation style',
    'a {0} in minimalism style',
    'a {0} in charcoal drawing style'
]

position_set = [
    'a {0} lying in the bed',
    'a {0} standing',
    'a {0} sitting on the chair',
    'a {0} jumping on the trampoline',
    'a {0} photo in profile',
    'a {0} swimming ',
    'a {0} sleeping ',
    'a {0} dancing',
    'a {0} running on a treadmill',
    'a {0} riding a bike',
    'a {0} skateboarding',
    'a {0} riding a horse',
    'a {0} riding a surfboard',
    'a {0} jumping with a parachute',
    'a {0} doing yoga',
    'a {0} sweeping the floor',
    'a {0} as an acrobat',
    'a {0} doing gymnastics',
    'a closed up photo of a {0}',
    'a distant photo of a {0}'
]

base_set = [
    "a photo of a {0}",
    # "a photo of the {0}",
    # "a photo of one {0}",
    # "a photo of my {0}",
    # "a rendering of a {0}",
    # "a rendering of the {0}",
    # "a cropped photo of a {0}",
    # "a cropped photo of the {0}",
    # "a photo of a cool {0}",
    # "a photo of the cool {0}",
    # "a bright photo of a {0}",
    # "a bright photo of the {0}",
    # "a dark photo of a {0}",
    # "a dark photo of the {0}",
    # "a rendition of a {0}",
    # "a rendition of the {0}",
    # "a photo of a nice {0}",
    # "a photo of the nice {0}",
    # "a good photo of a {0}",
    # "a good photo of the {0}",
]

base_set_clean = [
    "a photo of a {0}",
    # "a photo of the {0}",
    "a photo of one {0}",
    # "a photo of my {0}",
    # "a rendering of a {0}",
    # "a rendering of the {0}",
    "a cropped photo of a {0}",
    # "a cropped photo of the {0}",
    # "a photo of a cool {0}",
    # "a photo of the cool {0}",
    # "a bright photo of a {0}",
    # "a bright photo of the {0}",
    # "a dark photo of a {0}",
    # "a dark photo of the {0}",
    # "a rendition of a {0}",
    # "a rendition of the {0}",
    # "a photo of a nice {0}",
    # "a photo of the nice {0}",
    # "a good photo of a {0}",
    # "a good photo of the {0}",
]

object_set = [
    'a {0} in the jungle',
    'a {0} in the snow',
    'a {0} on the beach',
    'a {0} on a cobblestone street',
    'a {0} on top of pink fabric',
    'a {0} on top of a wooden floor',
    'a {0} with a city in the background',
    'a {0} with a mountain in the background',
    'a {0} with a blue house in the background',
    'a {0} on top of a purple rug in a forest',
    'a {0} with a wheat field in the background',
    'a {0} with a tree and autumn leaves in the background',
    'a {0} with the Eiffel Tower in the background',
    'a {0} floating on top of water',
    'a {0} floating in an ocean of milk',
    'a {0} on top of green grass with sunflowers around it',
    'a {0} on top of a mirror',
    'a {0} on top of the sidewalk in a crowded street',
    'a {0} on top of a dirt road',
    'a {0} on top of a white rug',
    'a red {0}',
    'a purple {0}',
    'a shiny {0}',
    'a wet {0}',
    'a cube shaped {0}'
]

live_set = [
    'a {0} in the jungle',
    'a {0} in the snow',
    'a {0} on the beach',
    'a {0} on a cobblestone street',
    'a {0} on top of pink fabric',
    'a {0} on top of a wooden floor',
    'a {0} with a city in the background',
    'a {0} with a mountain in the background',
    'a {0} with a blue house in the background',
    'a {0} on top of a purple rug in a forest',
    'a {0} wearing a red hat',
    'a {0} wearing a santa hat',
    'a {0} wearing a rainbow scarf',
    'a {0} wearing a black top hat and a monocle',
    'a {0} in a chef outfit',
    'a {0} in a firefighter outfit',
    'a {0} in a police outfit',
    'a {0} wearing pink glasses',
    'a {0} wearing a yellow shirt',
    'a {0} in a purple wizard outfit',
    'a red {0}',
    'a purple {0}',
    'a shiny {0}',
    'a wet {0}',
    'a cube shaped {0}'
]

merge_test_set = [
    # 'a dog in <style> style and <style2> style',
    # 'a dog in <style2> style',
    # 'a dog in <style> style',
    # 'a {0}',
    # 'a {0} in the jungle',
    # 'a {0} in the jungle in <style> style',
    # 'a {0} in the jungle in <style2> style',
    # 'a {0} wearing a hat in {1} style',
    # 'a {0} wearing a green bow tie in {1} style',
    # 'a {0} wearing a santa hat in {1} style',
    # 'a {0} as an astronaut in {1} style',
    # 'a {0} with a crown in {1} style',
    # 'a {0} in a helmet in {1} style',
    'a {0} riding a bicycle in {1} style',
    # 'a {0} riding a surfboard in {1} style',
    # 'a {0} doing yoga in {1} style',
    # 'a {0} sleeping in {1} style',
    # 'a {0} dancing in {1} style',
    # 'a {0} in a boat {1} style',
    # 'a {0} driving a car in {1} style',
    # 'a {0} skateboarding in {1} style',
    "a {0} in {1} style"
    # "a {0} in <style2> style"
]

merge_base_set = [
    # "a photo of a {0} in <style> style",
    # "a photo of a {0} in <style2> style",
    # "a photo of a {0} in {1} style",
    # "a {0} in {1} style",
]

# merge_test_set = [
#     'a {0} and <cat> in the jungle',
#     #'a {0} in the jungle in <style2> style',
# ]

# merge_base_set = [
#     "a photo of a {0} and <cat>",
#     #"a photo of a {0} in <style2> style",
# ]

large_set = appearance_set + outfit_set + background_set + style_set + position_set
small_set = [appearance_set[0], outfit_set[0], background_set[0], style_set[0], position_set[0]]
medium_set = appearance_set[:5] + outfit_set[:5] + background_set[:5] + style_set[:5] + position_set[:5]
medium_set_clean = appearance_set[:5] + background_set[:5] + style_set[:5]
augmentation_set = [
    'with the bookshelf in the background',
    'with Leaning Tower of Pisa in the background ',
    'in the bath',
    'with the river in the background',
    'on the moon',
    'on top of the blue fluffy carpet',
    'with the pool in the background',
    'with a gray brick wall in the background',
    'with a rainbow in the background',
    'with the playground in the background',
]

evaluation_sets = {
    '': [base_set[0]],
    'base': base_set,
    'large': large_set,
    'small': small_set,
    'medium': medium_set,
    'live': live_set,
    'object': object_set,
    'medium_clean': medium_set_clean,
    'augmentation': augmentation_set,

    'medium_appearance': appearance_set[:5],
    'medium_outfit': outfit_set[:5],
    'medium_background': background_set[:5],
    'medium_style': style_set[:5],
    'medium_position': position_set[:5],

    'joined': [prompt.replace('{0}', '{0}') for prompt in joined_set],
}
