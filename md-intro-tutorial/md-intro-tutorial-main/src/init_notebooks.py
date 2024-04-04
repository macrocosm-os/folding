from IPython.core.display import display, HTML
display(HTML(open("src/gromacs-training.css").read()))
import random

def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Show/hide solution'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += 'coming up'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )

    return HTML(html)

import json
import os.path
import re
import ipykernel
import requests

from requests.compat import urljoin
from notebook.notebookapp import list_running_servers

def get_notebook_name():
    """
    Return the full path of the jupyter notebook.
    """
    kernel_id = re.search('kernel-(.*).json',
                          ipykernel.connect.get_connection_file()).group(1)
    servers = list_running_servers()
    for ss in servers:
        response = requests.get(urljoin(ss['url'], 'api/sessions'),
                                params={'token': ss.get('token', '')})
        for nn in json.loads(response.text):
            if nn['kernel']['id'] == kernel_id:
                relative_path = nn['notebook']['path']
                return os.path.join(ss['notebook_dir'], relative_path)

def check_notebook():
    if os.getenv('JUPYTERHUB_API_TOKEN') is None:
        import hashlib
        notebook_name = get_notebook_name()
		
        # read in reference checksum
        nbpathhead, nbname = os.path.split(notebook_name)
        results = open(nbpathhead + '/src/.check/' + nbname + '.md5','r').read()
        checksum_reference = str(results.split()[0])

        # evaluate currrent checksum
        checksum_current = hashlib.md5(open(notebook_name,'rb').read()).hexdigest()

        #report
        if checksum_current == checksum_reference:
            print("Notebook is unchanged from source")
        else:
            print("Notebook has been modified")
    else:
        print("Notebook is running on JupyterHub so there's no need to check it for changes")


##Basic multiple-choice widget from https://levelup.gitconnected.com/deploy-simple-and-instant-online-quizzes-with-jupyter-notebook-tools-5e10f37da531

from ipywidgets import widgets, Layout, Box, GridspecLayout

def create_multipleChoice_widget(description, options, correct_answer, hint):
    if correct_answer not in options:
        options.append(correct_answer)
    
    correct_answer_index = options.index(correct_answer)
    
    radio_options = [(words, i) for i, words in enumerate(options)]
    alternativ = widgets.RadioButtons(
        options = radio_options,
        description = '',
        disabled = False,
        indent = False,
        align = 'center',
    )
    
    description_out = widgets.Output(layout=Layout(width='auto'))
    
    with description_out:
        print(description)
        
    feedback_out = widgets.Output()

    def check_selection(b):
        a = int(alternativ.value)
        if a==correct_answer_index:
            s = '\x1b[6;30;42m' + "correct" + '\x1b[0m' +"\n"
        else:
            s = '\x1b[5;30;41m' + "try again" + '\x1b[0m' +"\n"
        with feedback_out:
            feedback_out.clear_output()
            print(s)
        return
    
    check = widgets.Button(description="check")
    check.on_click(check_selection)
    
    hint_out = widgets.Output()
    
    def hint_selection(b):
        with hint_out:
            print(hint)
            
        with feedback_out:
            feedback_out.clear_output()
            print(hint)
    
    hintbutton = widgets.Button(description="hint")
    hintbutton.on_click(hint_selection)
    
    return widgets.VBox([description_out, 
                         alternativ, 
                         widgets.HBox([hintbutton, check]), feedback_out], 
                        layout=Layout(display='flex',
                                     flex_flow='column',
                                     align_items='stretch',
                                     width='auto'))

# Read xvg and use matplot lib, adapted from https://github.com/JoaoRodrigues/gmx-tools/blob/master/xvg_plot.py

import os, re, shlex
import matplotlib.pyplot as plt
import numpy as np

def parse_xvg(fname, sel_columns='all'):
    """Parses XVG file legends and data"""
    
    _ignored = set(('legend', 'view'))
    _re_series = re.compile('s[0-9]+$')
    _re_xyaxis = re.compile('[xy]axis$')

    metadata = {}
    num_data = []
    
    metadata['labels'] = {}
    metadata['labels']['series'] = []

    ff_path = os.path.abspath(fname)
    if not os.path.isfile(ff_path):
        raise IOError('File not readable: {0}'.format(ff_path))
    
    with open(ff_path, 'r') as fhandle:
        for line in fhandle:
            line = line.strip()
#            print(line)
            if line.startswith('@'):
                tokens = shlex.split(line[1:])
                if tokens[0] in _ignored:
                    continue
                elif tokens[0] == 'TYPE':
                    if tokens[1] != 'xy':
                        raise ValueError('Chart type unsupported: \'{0}\'. Must be \'xy\''.format(tokens[1]))
                elif _re_series.match(tokens[0]):
                    metadata['labels']['series'].append(tokens[-1])
                elif _re_xyaxis.match(tokens[0]):
                    metadata['labels'][tokens[0]] = tokens[-1]
                elif len(tokens) == 2:
                    metadata[tokens[0]] = tokens[1]
                else:
                    print('Unsupported entry: {0} - ignoring'.format(tokens[0]), file=sys.stderr)
            elif line[0].isdigit():
                num_data.append(list(map(float, line.split())))
    
    num_data = list(zip(*num_data))

    if not metadata['labels']['series']:
        for series in range(len(list(num_data)) - 1):
            metadata['labels']['series'].append('')

    # Column selection if asked
    if sel_columns != 'all':
        sel_columns = map(int, sel_columns)
        x_axis = num_data[0]
        num_data = [x_axis] + [num_data[col] for col in sel_columns]
        metadata['labels']['series'] = [metadata['labels']['series'][col - 1] for col in sel_columns]
    
    return metadata, num_data

def plot_data(data, metadata, window=1, interactive=True, outfile=None, 
              colormap='Set1', bg_color='lightgray'):
    """
    Plotting function.
    """

    n_series = len(data) - 1
    
    f = plt.figure()
    ax = plt.gca()
    
    color_map = getattr(plt.cm, colormap)
    color_list = color_map(np.linspace(0, 1, n_series))

    for i, series in enumerate(data[1:]):

        label = metadata['labels']['series'][i]
        
        # Adjust x-axis for running average series
        if label.endswith('(Av)'):
            x_step = (data[0][1] - data[0][0])
            x_window = (window * x_step) / 2
            x_start = data[0][0] + x_window - x_step
            x_end = data[0][-1] - x_window + x_step
            x_data = np.arange(x_start, x_end, x_step)
        else:
            x_data = data[0]
        
        ax.plot(x_data, series, c=color_list[i], label=label)

    # Formatting Labels & Appearance
    _re_reaction_coordinate = re.compile(r'\\xx\\f\{\}')
    xlabel = metadata['labels'].get('xaxis', '')
    xlabel = _re_reaction_coordinate.sub("Reaction Coordinate", xlabel) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metadata['labels'].get('yaxis', ''))
    ax.set_title(metadata.get('title', ''))
    
    ax.set_facecolor(bg_color)
    ax.grid('on')
    
    # try:
    #     legend = ax.legend()
    #     frame = legend.get_frame()
    #     frame.set_facecolor(bg_color)
    # except AttributeError as e:
    #     # No legend, likely because no labels
    #     pass
    
    if outfile:
        plt.savefig(outfile)
        
    if interactive:
        plt.show()
    
    return

from subprocess import Popen, PIPE, STDOUT

# Helper function for running a gmx command within a notebook,
# while piping the terminal output to the notebook.
def run_command(command):
    with Popen(command.split(), stdout=PIPE, stderr=STDOUT, universal_newlines=True, bufsize=1) as p:
        output = "".join([print(buf, end="") or buf for buf in p.stdout])


import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
try:
    import nglview as nv
except:
    print("This notebook requires nglview. Use e.g. pip install nglview, then restart this notebook")
