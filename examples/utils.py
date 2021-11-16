import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import numpy as np
from collections import namedtuple

from template import Template

ParticleProperties = namedtuple('ParticleProperties', 'd k density m gamma I')

def NewParticleProperties(d=1.0, k=1e6, rho=1.0, gamma=0.0):
    volume = 4./3. * np.pi * (d/2)**3
    m = volume * rho
    I = 2./5. * m * (d/2)**2
    return ParticleProperties(
        d=d,
        m=m,
        k=k,
        density=rho,
        gamma=gamma,
        I=I,
    )

def CacheMyDataFrame(cache_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            arg_hash = str(args)
            filename = cache_name + arg_hash + '.pkl'
            if os.path.exists(filename):
                print('[CACHE]: loading {0}'.format(filename))
                data = pd.read_pickle(filename)
            else:
                dirname = os.path.dirname(cache_name)
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)
                data = func(*args, **kwargs)
                print('[CACHE]: saving {0}'.format(filename))
                data.to_pickle(filename)
            return data

        return wrapper
    return decorator

def make_lammps_filename(prefix, ext, dt, gamma_n=None, dy=None, theta=None):
    dt_string = '{0:f}'.format(dt).rstrip('0')
    if gamma_n is not None:
        dt_string += '_{0:d}'.format(gamma_n)
    if dy is not None:
        dt_string += '_{0:f}'.format(dy).rstrip('0') if dy > 0 else '_0'
    if theta is not None:
        dt_string += '_{0:d}'.format(int(theta))
    filename = '{0}_{1}.{2}'.format(prefix, dt_string, ext)
    return filename

def check_and_load_lammps(filename):
    if os.path.exists(filename):
        return load_lammps_file(filename)
    print('Lammps file not found:', filename)
    return None

#def load_lammps_file(filename):
#    with open(filename, 'r') as f:
#        header = f.readline()
#        col_names = header.split(' ')
#        str_lines = [line.strip(' \n') for line in f.readlines()]
#        lines = [[float(num) for num in line.split(' ') if len(num) > 0]
#                 for line in str_lines]
#        n_cols = len(lines[0])
#        num_entries = len(str_lines)
#        frame = pd.DataFrame(lines, index=range(num_entries), columns=col_names[:n_cols])
#        return frame
#
#    filename = '{0}_{1}.{2}'.format(prefix, dt_string, ext)
#    print(filename)
#    return filename


def load_lammps_file(filename):
    with open(filename, 'r') as f:
        header = f.readline()
        col_names = header.split(' ')
        str_lines = [line.strip(' \n') for line in f.readlines()]
        lines = [[float(num) for num in line.split(' ') if len(num) > 0]
                 for line in str_lines]
        n_cols = len(lines[0])
        num_entries = len(str_lines)
        frame = pd.DataFrame(lines, index=range(num_entries), columns=col_names[:n_cols])
        return frame

def load_lammps_dump(filename, col_names=[]):

    def read_timestep(f):
        return int(f.readline())

    def read_num_entries(f):
        return int(f.readline())

    def read_bounds(f):
        bounds = (f.readline().split(' ') +
                  f.readline().split(' ') +
                  f.readline().split(' '))
        return [float(b) for b in bounds]

    def read_data(f, col_names, num_entries):

        str_lines = [f.readline().strip(' \n') for i in range(num_entries)]
        lines = [[float(num) for num in line.split(' ')]
                 for line in str_lines]
        n_cols = len(lines[0])
        frame = pd.DataFrame(lines, index=range(num_entries), columns=col_names[:n_cols])
        return frame

    frames = {}
    #print('read:',filename)
    with open(filename, 'r') as f:
        line = f.readline().strip('\n')
        while line:
            tokens = line.split(' ')
            if tokens[1] == 'TIMESTEP':
                timestep = read_timestep(f)
            elif tokens[1] == 'NUMBER':
                num_entries = read_num_entries(f)
            elif tokens[1] == 'BOX':
                bounds = read_bounds(f)
            elif tokens[1] == 'ATOMS':
                data = read_data(f, tokens[2:], num_entries)
                frames[timestep] = data
            elif tokens[1] == 'ENTRIES':
                data = read_data(f, col_names or tokens[2:], num_entries)
                frames[timestep] = data
            else:
                print('Unknown LAMMPS dump format starting with:', tokens[:2])
                print(tokens)
            line = f.readline().strip(' \n')

    return frames



class Figure:

    def __init__(self, xlabel, ylabel, show=True, tikz_filename=None, template_filename=None, dat_filename=None):
        self.show = show

        if show:
            self.fig = plt.figure()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            self.xlabel = xlabel
            self.ylabel = ylabel

            self.dat_filename = dat_filename
            self.data = [] if not dat_filename is None else None
            self.n_plot = 0
            self.n_colors = 0

            self.tikz_filename = tikz_filename
            self.template_filename = template_filename

            self.plots = []

        else:
            self.fig = None



    def __del__(self):
        if not self.show:
            return

        if self.dat_filename:
            for i, d in enumerate(self.data):
                d.to_csv(self.dat_filename.format(i+1), sep='\t', index=False)

        if self.template_filename:
            template = Template(self.template_filename)

            ## Substitute the color scheme for each of the plots
            n_color_scheme = max(self.n_colors, 3)
            self.plots = [
                plot % (n_color_scheme)
                for plot in self.plots
            ]

            ## Render the tex template
            tex = template.render(
                colorscheme='Dark2-%d' % (n_color_scheme),
                xlabel=self.xlabel,
                ylabel=self.ylabel,
                plots=self.plots,
            )

            ## Save to text file if a filename is given, otherwise just print
            if self.tikz_filename:
                with open(self.tikz_filename, 'w') as out:
                    out.write(tex)

                self.write_deps()
            else:
                print(tex)

    def write_deps(self):
        '''
        Write a file describing the dependencies of the final pdf that can
        be imported into makefiles.
        '''

        _, ext = os.path.splitext(self.tikz_filename)
        dir_ = os.path.dirname(self.tikz_filename)
        deps_filename = self.tikz_filename.replace(ext, '.deps')
        pdf_filename = self.tikz_filename.replace(ext, '.pdf')

        with open(deps_filename, 'w') as out:
            if not dir_ == '':
                out.write(os.path.basename(pdf_filename))
                out.write(' : ')
                out.write(pdf_filename)
                out.write('\n')
            out.write(pdf_filename)
            out.write(' : ')
            out.write(self.tikz_filename)
            for i in range(self.n_plot):
                out.write(' ')
                out.write(self.dat_filename.format(i+1))
            out.write('\n')

    def plot(self, x, y, legend, **kwargs):
        if not self.show:
            return

        plt.figure(self.fig.number)
        plt.plot(x, y, legend, **kwargs)

        if 'label' in kwargs.keys():
            plt.legend(loc='best')

        self.n_plot += 1

        if self.dat_filename:
            #if self.n_plot == 1:
            #    self.data['x'] = x

            #self.data['y{0}'.format(self.n_plot)] = y #(x, y)
            d = pd.DataFrame()
            d['x'] = x
            d['y'] = y
            self.data.append(d)

            color_i = (int(legend[1])
                       if legend[1] in '0123456789'
                       else 0)
            self.n_colors = max(color_i+1, self.n_colors)
            legend_str = ('\n\t\t' + r'\addlegendentry{%s};' % (kwargs['label']) + '\n'
                          if 'label' in kwargs.keys()
                          else '')
            line_spec = ''
            if legend[2:4] == '--':
                line_spec = ', dashed'
            if legend[2:4] == '-.':
                line_spec = ', dash dot'
            plot_line = r'\addplot[index of colormap=%d of Dark2-%%d%s]table[x=x, y=y]{%s};%s' % (
                color_i,
                line_spec,
                self.dat_filename.format(self.n_plot),
                legend_str,
            )
            self.plots.append(plot_line)

def save_tikz_cartoon(filename, particles, drawvel=False, bonds=None, trails=None, tikzpicture_scale=1, p_scale=1, x_coord_idx=0, y_coord_idx=1, triangulation=None, extra_cmd=[]):

    N_steps_in_trail = 20
    if not trails is None:
        N_particles, N_t_steps = trails.shape
    else:
        N_particles = particles.N

    with open(filename, 'w') as out:
        out.write(r'\documentclass{standalone}' + '\n')
        out.write(r'\input{drawparticle.tex}' + '\n')
        out.write(r'\begin{document}' + '\n')
        out.write(r'\begin{tikzpicture}[scale=%f]' % tikzpicture_scale + '\n')

        z_coord_idx = (-x_coord_idx - y_coord_idx) % 3
        plot_order = np.argsort(particles.xyz[:, z_coord_idx])

        for i_part in plot_order: #range(N_particles):
            out.write(r'  \coordinate (pos{0}) at ({1}, {2});'.format(
                i_part,
                particles.xyz[i_part, x_coord_idx],
                particles.xyz[i_part, y_coord_idx],
            ) + '\n')
            out.write(r'  \drawparticle{pos%i}{%f}{%s};' % (
                i_part,
                particles.d/2 if type(particles.d) == float else particles.d[i_part]/2,
                'red!60' if particles.is_rep[i_part] else 'black!10'
            ) + '\n')
            if drawvel:
                out.write(r'  \coordinate (p{0}) at ({1}, {2});'.format(
                    i_part,
                    particles.d/4 * particles.p[i_part, x_coord_idx] / p_scale,
                    particles.d/4 * particles.p[i_part, y_coord_idx] / p_scale,
                ) + '\n')
                out.write(r'  \drawvel{pos%i}{p%i}' % (i_part, i_part) + '\n')

            if not trails is None:
                idx = np.arange(
                    0,
                    N_t_steps,
                    np.floor(N_t_steps/N_steps_in_trail),
                    dtype=np.int,
                )
                out.write(r'\draw')
                for i_idx in idx[:-1]:
                    out.write(r'    ({0}, {1}) --'.format(
                        pos_x[i_part, i_idx],
                        pos_y[i_part, i_idx],
                    ))
                out.write(r'    ({0}, {1});'.format(
                    pos_x[i_part, idx[-1]],
                    pos_y[i_part, idx[-1]],
                   ) + '\n')

        if not bonds is None:
            for i in range(N_particles):
                for j in range(i):
                    if j in bonds[i]:
                        out.write(r'  \drawbond{pos%i}{pos%i};\n' % (i, j) + '\n')

        if not triangulation is None:

            for i, cell in enumerate(triangulation.cells):
                for j in range(3):
                    out.write(
                        r'\coordinate (tri-%i-%i) at (%f, %f);' % (
                            i,
                            j,
                            cell.vertices[j, x_coord_idx],
                            cell.vertices[j, y_coord_idx],
                        )
                        + '\n'
                    )
            for i, cell in enumerate(triangulation.cells):
                for j in range(3):
                    out.write(
                        r'\drawtri{tri-%i-%i}{tri-%i-%i};' % (
                            i,
                            j,
                            i,
                            (j+1)%3
                        )
                        + '\n'
                    )
                    #lines[6*i + 2*j  ] = cell.vertices[j      , :]
                    #lines[6*i + 2*j+1] = cell.vertices[(j+1)%3, :]
        out.write('\n'.join(extra_cmd) + '\n')
        out.write(r'\end{tikzpicture}' + '\n')
        out.write(r'\end{document}' + '\n')
