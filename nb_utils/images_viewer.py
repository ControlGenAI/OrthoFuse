import os
import contextlib
from multiprocessing.pool import ThreadPool
from typing import Callable, Optional, Union, List, Tuple

import glob

import tqdm
import tqdm.autonotebook
from PIL.Image import Image

import numpy as np

import ipywidgets
from ipyfilechooser import FileChooser
from IPython.display import display, clear_output, Latex

import matplotlib.pyplot as plt

from .configs import _LOAD_IMAGE_BACKEND
from .utils import prepare_axes, _save_get, RegexFilter


def _display_widget(_, output, widget):
    output.clear_output()
    with output:
        display(widget)


def create_buttons_grid(grid_size):
    buttons_grid = ipywidgets.VBox([
        ipywidgets.HBox([
            ipywidgets.Button(
                description=f'({idx}, {jdx}) -> {idx * grid_size[1] + jdx}',
                layout=ipywidgets.Layout(width='{0}%'.format(100 / grid_size[1]))
            )
            for jdx in range(grid_size[1])
        ]) for idx in range(grid_size[0])
    ])

    buttons = [
        buttons_grid.children[idx].children[jdx]
        for idx in range(grid_size[0])
        for jdx in range(grid_size[1])
    ]
    return buttons_grid, buttons


class MultifolderViewer:
    # noinspection PyUnresolvedReferences
    def __init__(
            self, directories: Union[List[str], str], labels: Optional[List[str]] = None,
            lazy_load: bool = True, info=None, set_name='medium', filter_fn: RegexFilter = None
    ):
        """
        :param Union[List[str], str] directories: list of directories or glob template to display
        :param Optional[List[str]] labels: list of labels for each directory. If None then use directory's basename
        :param bool lazy_load: Whether to load all images in class constructor
        :param info:
        :param set_name
        :param RegexFilter filter_fn: Filter directories w.r.t. their labels.
            Use a .normalize method to determine order of directories
        """
        self.info = info
        self.set_name = set_name
        if isinstance(directories, str):
            assert labels is None
            directories = sorted([
                path
                for path in glob.glob(os.path.join(directories, '*'))
                if os.path.isdir(path) and not path.endswith('lmdb')
            ])

        if labels is None:
            labels = [os.path.basename(_) for _ in directories]
        else:
            assert len(labels) == len(directories)

        if len(labels):
            sort_fn = lambda x: x[1]

            if filter_fn is not None:
                sort_fn = lambda x: filter_fn.normalize(x[1])
                directories, labels = zip(*filter(lambda x: filter_fn(x[1]), zip(directories, labels)))
            directories, labels = zip(*sorted(zip(directories, labels), key=sort_fn))

        self.labels = labels
        self.directories = directories
        self._label_to_directory = dict(zip(self.labels, self.directories))

        self.images = {}
        self.images_paths = {}

        if not lazy_load:
            with ThreadPool(processes=4) as pool:
                def _load_wrapper(args):
                    _label, _directory = args
                    return _label, self.load_images(_directory)

                for label, (paths, images) in tqdm.autonotebook.tqdm(
                        pool.imap(_load_wrapper, zip(self.labels, self.directories)), total=len(self.directories)
                ):
                    self.images_paths[label], self.images[label] = paths, images

    @staticmethod
    def load_images(directory: str) -> Tuple[List[str], List[Image]]:
        """
        :param str directory: target path to the folder with images
        :return: list of paths and list of corresponding images
        """
        paths = sorted(glob.glob(os.path.join(directory, '*')))
        images = [_LOAD_IMAGE_BACKEND(path) for path in paths]

        return paths, images

    def _show_random(self, output, ncolumns=3, save_path=None):
        output = contextlib.nullcontext() if output is None else output

        def _show(_):
            with output:
                clear_output(wait=True)

                grid_size = np.array([(len(self.directories) - 1) // ncolumns + 1, ncolumns])
                fig, axes = plt.subplots(*grid_size, figsize=2 * grid_size[::-1])
                prepare_axes(axes)

                for ax, label in tqdm.autonotebook.tqdm(
                        zip(axes.reshape(-1), self.labels), total=len(self.labels), leave=False
                ):
                    if label not in self.images:
                        directory = self._label_to_directory[label]
                        self.images_paths[label], self.images[label] = self.load_images(directory)

                    images = self.images[label]
                    [idx] = np.random.choice(len(images), 1)
                    ax.imshow(images[idx])

                    if self.info is not None:
                        top_label, bottom_label = _get_random_image_annotations(self.info, label, idx)

                        ax.annotate(
                            top_label, xy=(0.01, 0.95),
                            xycoords='axes fraction', fontsize=6,
                            bbox=dict(boxstyle="round,pad=0.,rounding_size=0.01", alpha=0.5, color='w')
                        )
                        ax.annotate(
                            label, xy=(0.01, 0.885),
                            xycoords='axes fraction', fontsize=6,
                            bbox=dict(boxstyle="round,pad=0.,rounding_size=0.01", alpha=0.5, color='w')
                        )
                        ax.annotate(
                            os.path.basename(self.images_paths[label][idx]), xy=(0.01, 0.079),
                            xycoords='axes fraction', fontsize=6,
                            bbox=dict(boxstyle="round,pad=0.,rounding_size=0.01", alpha=0.5, color='w')
                        )
                        ax.annotate(
                            bottom_label, xy=(0.01, 0.017),
                            xycoords='axes fraction', fontsize=6,
                            bbox=dict(boxstyle="round,pad=0.,rounding_size=0.01", alpha=0.5, color='w')
                        )
                    else:
                        ax.annotate(
                            label, xy=(0.04, 0.9),
                            xycoords='axes fraction', fontsize=8,
                            bbox=dict(boxstyle="round", alpha=0.7, color='w')
                        )

                if self.info is not None:
                    subtitle = _get_random_figure_title(self.info, set_name=self.set_name)
                    fig.suptitle(subtitle, fontsize=14, y=0.9)
                    display(Latex(subtitle))
                fig.subplots_adjust(wspace=0, hspace=0)
                fig.patch.set_visible(False)

                fc = FileChooser('./', select_desc='Select path to save image')
                fc.default_filename = '*.jpg'

                def _save_image(chooser):
                    _save_path = os.path.join(chooser.selected)
                    if not os.path.exists(_save_path):
                        fig.savefig(_save_path, bbox_inches='tight', dpi=600, pad_inches=0)
                    else:
                        print(f'{_save_path} already exists')

                if save_path is not None:
                    fig.savefig(save_path, bbox_inches='tight', dpi=600, pad_inches=0)
                    fig.clf()
                    plt.cla()
                    plt.clf()
                else:
                    fc.register_callback(_save_image)
                    display(fc)
                    plt.show()

        return _show

    def _show_class(self, output, label, ncolumns, save_path=None):
        output = contextlib.nullcontext() if output is None else output

        def _show(_):
            with output:
                clear_output(wait=True)

                if label not in self.images:
                    directory = self._label_to_directory[label]
                    self.images_paths[label], self.images[label] = self.load_images(directory)

                images = self.images[label]
                paths = self.images_paths[label]

                grid_size = np.array([(len(images) - 1) // ncolumns + 1, ncolumns])
                fig, axes = plt.subplots(*grid_size, figsize=2 * grid_size[::-1])
                prepare_axes(axes)

                print(os.path.split(paths[0])[0])
                folder_name = os.path.normpath(paths[0]).split(os.path.sep)[-2]

                if self.info is not None:
                    title = _get_class_figure_title(self.info, folder_name, set_name=self.set_name)
                else:
                    title = label

                for idx, (ax, image, path) in enumerate(zip(axes.reshape(-1), images, paths)):
                    image_name = os.path.basename(path)

                    if self.info is not None:
                        image_title = _get_image_title(self.info, folder_name, image_name, idx)
                    else:
                        image_title = image_name

                    ax.imshow(image)
                    ax.annotate(
                        image_title, xy=(0.025, 0.95),
                        xycoords='axes fraction', fontsize=5,
                        bbox=dict(boxstyle="round", alpha=0.5, color='w')
                    )

                fig.suptitle(title)
                fig.subplots_adjust(wspace=0, hspace=0)

                fc = FileChooser('./', select_desc='Select path to save image')
                fc.default_filename = '*.jpg'

                def _save_image(chooser):
                    _save_path = os.path.join(chooser.selected)
                    if not os.path.exists(_save_path):
                        fig.savefig(_save_path, bbox_inches='tight', dpi=600, pad_inches=0)
                    else:
                        print(f'{_save_path} already exists')

                if save_path is not None:
                    fig.savefig(save_path, bbox_inches='tight', dpi=600, pad_inches=0)
                    fig.clf()
                    plt.cla()
                    plt.clf()
                else:
                    fc.register_callback(_save_image)
                    display(fc)
                    plt.show()

        return _show

    def view(self, ncolumns: int = 3) -> ipywidgets.Widget:
        """Draw widget (in form of buttons grid) where each button shows images from the corresponding folder
        :param int ncolumns: number of columns in this buttons grid
        :return:
        """
        grid_size = np.array([(len(self.directories) - 1) // ncolumns + 1, ncolumns])
        save_path = None

        output = ipywidgets.Output()

        rnd_button = ipywidgets.Button(description='Random Images', layout=ipywidgets.Layout(width='auto'))
        rnd_button.on_click(self._show_random(output, ncolumns=5, save_path=save_path))

        buttons_grid, buttons = create_buttons_grid(grid_size)
        for button, label in zip(buttons, self.labels):
            button.description = label
            button.on_click(self._show_class(output, label, ncolumns=ncolumns))

        selector = ipywidgets.VBox([
            rnd_button,
            buttons_grid,
            output
        ])

        return selector


def _get_random_figure_title(info, set_name: str):
    title_values = []
    for key in [
        'real_image_similarity', 'image_similarity', 'with_class_image_similarity',
        f'{set_name}_image_similarity', f'{set_name}_with_class_image_similarity',

        'dino_real_image_similarity', 'dino_image_similarity', 'dino_with_class_image_similarity',
        f'dino_{set_name}_image_similarity', f'dino_{set_name}_with_class_image_similarity',

        f'{set_name}_text_similarity', f'{set_name}_with_class_text_similarity',
        f'{set_name}_text_similarity_with_class', f'{set_name}_with_class_text_similarity_with_class',
    ]:
        value = _save_get(key, info)
        title_values.append('-' if value is None else '${0:.3f}$'.format(value))
    title = '$IS^{{R}}$/$IS^{{1}}$/$IS^{{1}}_{{wc}}$/$IS$/$IS_{{wc}}$: {}/{}/{}/{}/{} ({}/{}/{}/{}/{}), $TS$: {}/{}/{}/{}'.format(
        *title_values
    )

    return title


def _get_random_image_annotations(info, prompt: str, image_idx: int):
    top_values = []
    for key in [
        ('image_similarities', prompt), ('dino_image_similarities', prompt),
        ('text_similarities', prompt), ('text_similarities_with_class', prompt)
    ]:
        value = _save_get(key, info)
        top_values.append('-' if value is None else '{0:.2f}'.format(value))
    top_label = '$IS$: {} ({}), $TS$/$TS_{{wc}}$: {}/{}'.format(*top_values)

    bottom_values = []
    value = _save_get(('image_similarities_mx', prompt), info)
    bottom_values.append(
        '-' if value is None else '{0:.2f}'.format(np.mean(np.array(value)[:, image_idx]))
    )

    value = _save_get(('dino_image_similarities_mx', prompt), info)
    bottom_values.append(
        '-' if value is None else '{0:.2f}'.format(np.mean(np.array(value)[:, image_idx]))
    )

    value = _save_get(('text_similarities_mx', prompt), info)
    bottom_values.append('-' if value is None else '{0:.2f}'.format(value[0][image_idx]))

    value = _save_get(('text_similarities_mx_with_class', prompt), info)
    bottom_values.append('-' if value is None else '{0:.2f}'.format(value[0][image_idx]))

    bottom_label = '$IS$: {} ({}), $TS$/$TS_{{wc}}$: {}/{}'.format(*bottom_values)

    return top_label, bottom_label


def _get_class_figure_title(info, prompt: str, set_name: str):
    title_values = []
    for key in [
        'real_image_similarity', 'image_similarity', 'with_class_image_similarity_',
        f'{set_name}_image_similarity', f'{set_name}_with_class_image_similarity',

        'dino_real_image_similarity', 'dino_image_similarity', 'dino_with_class_image_similarity',
        f'dino_{set_name}_image_similarity', f'dino_{set_name}_with_class_image_similarity',

        ('image_similarities', prompt), ('dino_image_similarities', prompt),

        ('text_similarities', prompt), ('text_similarities_with_class', prompt)
    ]:
        value = _save_get(key, info)
        title_values.append('-' if value is None else '${0:.3f}$'.format(value))

    title = (
        '$IS^{{R}}$/$IS^{{1}}$/$IS^{{1}}_{{wc}}$/$IS$/$IS_{{wc}}$: {}/{}/{}/{}/{} ({}/{}/{}/{}/{})\n'
        '$IS$: {} ({}), $TS$/$TS_{{wc}}$: {}/{} {}'
    ).format(
        *title_values, prompt
    )

    return title


def _get_image_title(info, prompt: str, image_name: str, image_idx: int):
    image_title_values = [image_name]

    value = _save_get(('image_similarities_mx', prompt), info)
    image_title_values.append(
        '-' if value is None else '{0:.2f}'.format(np.mean(np.array(value)[:, image_idx]))
    )

    value = _save_get(('dino_image_similarities_mx', prompt), info)
    image_title_values.append(
        '-' if value is None else '{0:.2f}'.format(np.mean(np.array(value)[:, image_idx]))
    )

    value = _save_get(('text_similarities_mx', prompt), info)
    image_title_values.append('-' if value is None else '{0:.2f}'.format(value[0][image_idx]))

    value = _save_get(('text_similarities_mx_with_class', prompt), info)
    image_title_values.append('-' if value is None else '{0:.2f}'.format(value[0][image_idx]))

    image_title = '{0}, $IS$: {1} ({2}), $TS$/$TS_{{wc}}$: {3}/{4}'.format(*image_title_values)

    return image_title
