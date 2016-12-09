from __future__ import absolute_import
import os
import logging
import numpy as np
import scipy.stats
from scipy import odr
from tornado.escape import json_encode

from .base import BaseHandler


class CompositionPlotHandler(BaseHandler):
  def get(self, fignum):
    '''Downloads plot data as text.'''
    fig_data = self.get_fig_data(int(fignum))
    ds = self.get_dataset(self.get_argument('ds_kind'),
                          self.get_argument('ds_name'))
    if fig_data is None or ds is None or fig_data.last_plot != 'compositions':
      self.write('Oops, something went wrong. Try again?')
      return

    ax = fig_data.figure.gca()
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()

    if ax.collections:
      pc, = ax.collections
      xy_data = pc.get_offsets()
      kind = 'xy'
    else:
      xy_data = fig_data.hist_data
      kind = 'x' if xlabel else 'y'

    title = ax.get_title()
    if title:
      xlabel = title + ': ' + xlabel
      ylabel = title + ': ' + ylabel

    if ds.kind != 'LIBS' or ds.name != 'MSL ChemCam':
      self.write('Downloading spectrum data for non-ChemCam data is NYI.')
      return

    # TODO: remove ChemCam-specific stuff here
    mask = fig_data.filter_mask[ds]
    sols = ds.metadata['sol'].get_array(mask)
    locs = ds.metadata['loc'].get_array(mask)
    shots = ds.metadata['shot'].get_array(mask)
    targets = ds.metadata['target'].get_array(mask)

    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition',
                    'attachment; filename='+fname)
    if kind == 'xy':
      self.write('Target Name,Sol,Location,Shot,%s,%s\n' % (xlabel, ylabel))
      for i, (x, y) in enumerate(xy_data):
        self.write('%s,%s,%s,%s,%g,%g\n' % (targets[i], sols[i], locs[i],
                                            shots[i], x, y))
    else:
      labels = [t.get_text() for t in ax.legend_.texts]
      self.write('Target Name,Sol,Location,Shot,%s\n' % ','.join(labels))
      for i, t in enumerate(targets):
        vals = ','.join('%g' % data[i] for data in xy_data)
        self.write('%s,%s,%s,%s,%s\n' % (t, sols[i], locs[i], shots[i], vals))
    self.finish()

  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      return self.visible_error(403, 'Broken connection to server.')

    ds = self.get_dataset(self.get_argument('ds_kind'),
                          self.get_argument('ds_name'))
    if ds is None:
      return self.visible_error(404, 'Failed to look up dataset.')

    do_fit = bool(int(self.get_argument('do_fit')))
    use_mols = bool(int(self.get_argument('use_mols')))
    x_input = self.get_argument('x_comps')
    y_input = self.get_argument('y_comps')
    if (not x_input) and (not y_input):
      return
    elif (not x_input) or (not y_input):
      do_fit = False
    # input has the form: 'comp:element+comp:element'
    x_keys = [k.split('$',1) for k in x_input.split('+')] if x_input else []
    y_keys = [k.split('$',1) for k in y_input.split('+')] if y_input else []
    use_group_name = len(set(k[0] for k in (x_keys + y_keys))) > 1
    do_sum = x_keys and y_keys
    mask = fig_data.filter_mask[ds]
    x_data, x_labels = comps_with_labels(ds, mask, x_keys, use_group_name,
                                         use_mols, do_sum)
    y_data, y_labels = comps_with_labels(ds, mask, y_keys, use_group_name,
                                         use_mols, do_sum)
    # error handling
    if x_data is None:
      return self.visible_error(403, x_labels)
    if y_data is None:
      return self.visible_error(403, y_labels)

    if do_fit:
      # handle NaNs
      notnan_mask = ~(np.isnan(x_data) | np.isnan(y_data))
      x_data = x_data[notnan_mask]
      y_data = y_data[notnan_mask]
      mask[mask] = notnan_mask

      logging.info('Running ODR on %d pairs of compositions: %s vs %s',
                   x_data.shape[0], x_labels, y_labels)

      # compute line of best fit
      rval, _ = scipy.stats.pearsonr(x_data, y_data)
      model = odr.ODR(odr.Data(x_data, y_data), odr.unilinear)
      result = model.run()
      slope, yint = result.beta
      slope_err, yint_err = result.sd_beta
      xint = -yint / slope
      xint_err = np.abs(xint) * np.linalg.norm([yint_err/yint, slope_err/slope])
      results = dict(rval=rval**2, slope=slope, slope_err=slope_err,
                     yint=yint, yint_err=yint_err, xint=xint, xint_err=xint_err)

      # make coordinates for the fit line
      x_min, x_max = x_data.min(), x_data.max()
      x_range = x_max - x_min
      fit_x = np.array([x_min - 0.25 * x_range, x_max + 0.25 * x_range])
      fit_y = fit_x * slope + yint
    else:
      results = {}

    fig_data.figure.clf(keep_observers=True)
    ax = fig_data.figure.gca()

    # setup plot options
    if bool(int(self.get_argument('legend'))):
      legend_loc = 'best'
    else:
      legend_loc = (2, 2)  # move legend off the figure entirely
    cmap = self.get_argument('cmap')
    scatter_kwargs = dict(
        alpha=float(self.get_argument('alpha')),
        cmap=(cmap if cmap != '_auto' else None),
        s=20*float(self.get_argument('line_width')))

    # set plot title (if needed)
    if not use_group_name:
      key, _ = (x_keys if x_keys else y_keys)[0]
      ax.set_title(ds.metadata[key].display_name(key))

    if do_sum:
      # scatter plot: x vs y
      color_key = self.get_argument('color_by')
      color_meta = ds.metadata.get(color_key, None)
      if color_meta is not None:
        colors = color_meta.get_array(mask)
        sc = ax.scatter(x_data, y_data, c=colors, **scatter_kwargs)
        cbar = fig_data.figure.colorbar(sc)
        cbar.set_label(color_meta.display_name(color_key))
      else:
        ax.scatter(x_data, y_data, **scatter_kwargs)
      suffix = ' (moles)' if use_mols else ''
      ax.set_xlabel(' + '.join(x_labels) + suffix)
      ax.set_ylabel(' + '.join(y_labels) + suffix)
    elif x_keys:
      # histogram along x
      ax.hist(x_data, bins='auto', orientation='vertical', label=x_labels)
      fig_data.hist_data = x_data
      ax.legend(loc=legend_loc)
    else:
      # histogram along y
      ax.hist(y_data, bins='auto', orientation='horizontal', label=y_labels)
      fig_data.hist_data = y_data
      ax.legend(loc=legend_loc)

    # get the plot bounds
    ax.autoscale_view()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # plot the best fit line (optionally)
    if do_fit:
      ax.plot(fit_x, fit_y, 'k--')
      ax.set_xlim(xlim)
      ax.set_ylim(ylim)

    # draw!
    fig_data.manager.canvas.draw()
    fig_data.last_plot = 'compositions'

    # respond with fit parameters and zoom info
    results['zoom'] = (xlim[0], xlim[1], ylim[0], ylim[1])
    return self.write(json_encode(results))

"""
class CompositionBatchHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      return
    ds = DATASETS['LIBS']['MSL ChemCam']
    use_mols = bool(int(self.get_argument('use_mols')))
    x_input = self.get_argument('x_comps')
    y_input = self.get_argument('y_comps')
    if (not x_input) or (not y_input):
      return
    # input has the form: 'comp:element+comp:element'
    x_keys = [k.split('$',1) for k in x_input.split('+')]
    y_keys = [k.split('$',1) for k in y_input.split('+')]
    use_group_name = len(set(k[0] for k in (x_keys + y_keys))) > 1

    # compute ratio over entire dataset
    x_data, x_labels = comps_with_labels(ds, Ellipsis, x_keys, use_group_name,
                                         use_mols)
    y_data, y_labels = comps_with_labels(ds, Ellipsis, y_keys, use_group_name,
                                         use_mols)

    # group by target+location+sol
    # TODO

    # run ODR on each group independently
    # TODO

    # filter results by the passed arguments
    # TODO
"""


def comps_with_labels(ds, mask, comp_keys, use_group_name=True, use_mols=True,
                      do_sum=True):
  tmp, labels = [], []
  for k1, k2 in comp_keys:
    m1 = ds.metadata[k1]
    m2 = m1.comps[k2]
    name = m2.display_name(k2)
    if use_mols:
      if name not in MOL_DATA:
        return None, 'Mol conversion unavailable for ' + name
      factor, name = MOL_DATA[name]
    else:
      factor = 1
    if use_group_name:
      name = '(%s: %s)' % (m1.display_name(k1), name)
    tmp.append((m2, factor))
    labels.append(name)

  convert_fn = sum if do_sum else list
  data = convert_fn(f * meta.get_array(mask) for meta, f in tmp)
  return data, labels

MOL_DATA = {
  # Oxides
  'Al2O3': (0.01962, 'Al'),
  'CaO': (0.01783, 'Ca'),
  'Fe2O3': (0.012524, 'Fe'),
  'FeOT': (0.01392, 'Fe'),
  'K2O': (0.02123, 'K'),
  'MgO': (0.02481, 'Mg'),
  'MnO': (0.01410, 'Mn'),
  'Na2O': (0.03227, 'Na'),
  'SiO2': (0.01664, 'Si'),
  'TiO2': (0.01252, 'Ti'),
  'CO2': (0.0227, 'C'),
  'H2O': (0.1110, 'H'),
  'SO3': (0.0125, 'S'),
  # Trace elements
  'Ag': (1e-4/107.8682, 'Ag'),
  'As': (1e-4/74.9216, 'As'),
  'Au': (1e-4/196.96655, 'Au'),
  'B': (1e-4/10.81, 'B'),
  'Ba': (1e-4/137.327, 'Ba'),
  'Be': (1e-4/9.012, 'Be'),
  'Bi': (1e-4/208.98040, 'Bi'),
  'Br': (1e-4/79.904, 'Br'),
  'Cd': (1e-4/112.414, 'Cd'),
  'Ce': (1e-4/140.116, 'Ce'),
  'Cl': (1/35.4527, 'Cl'),
  'Co': (1e-4/58.9332, 'Co'),
  'Cr': (1e-4/51.9961, 'Cr'),
  'Cs': (1e-4/132.90545, 'Cs'),
  'Cu': (1e-4/63.546, 'Cu'),
  'Dy': (1e-4/162.5, 'Dy'),
  'Er': (1e-4/67.26, 'Er'),
  'Eu': (1e-4/151.964, 'Eu'),
  'F': (1/18.9984023, 'F'),
  'Ga': (1e-4/69.723, 'Ga'),
  'Gd': (1e-4/157.25, 'Gd'),
  'Ge': (1e-4/72.61, 'Ge'),
  'Hf': (1e-4/178.49, 'Hf'),
  'Hg': (1e-4/200.592, 'Hg'),
  'Ho': (1e-4/164.93032, 'Ho'),
  'I': (1e-4/126.90447, 'I'),
  'In': (1e-4/114.818, 'In'),
  'Ir': (1e-4/192.21, 'Ir'),
  'La': (1e-4/138.9055, 'La'),
  'Li': (1e-4/6.941, 'Li'),
  'Lu': (1e-4/174.967, 'Lu'),
  'Mn': (1e-4/54.938049, 'Mn'),
  'Mo': (1e-4/95.94, 'Mo'),
  'N': (1e-4/14.007, 'N'),
  'Nb': (1e-4/92.906, 'Nb'),
  'Nd': (1e-4/144.24, 'Nd'),
  'Ni': (1e-4/58.6934, 'Ni'),
  'P': (1e-4/30.973762, 'P'),
  'Pb': (1e-4/207.2, 'Pb'),
  'Pd': (1e-4/106.42, 'Pd'),
  'Pr': (1e-4/140.90765, 'Pr'),
  'Pt': (1e-4/195.084, 'Pt'),
  'Rb': (1e-4/85.4678, 'Rb'),
  'Sb': (1e-4/121.76, 'Sb'),
  'Sc': (1e-4/44.95591, 'Sc'),
  'Se': (1e-4/78.971, 'Se'),
  'Sm': (1e-4/150.36, 'Sm'),
  'Sn': (1e-4/118.71, 'Sn'),
  'Sr': (1e-4/87.62, 'Sr'),
  'Ta': (1e-4/180.9479, 'Ta'),
  'Tb': (1e-4/58.92534, 'Tb'),
  'Te': (1e-4/127.60, 'Te'),
  'Th': (1e-4/232.0381, 'Th'),
  'Tl': (1e-4/47.867, 'Tl'),
  'Tm': (1e-4/168.93421, 'Tm'),
  'U': (1e-4/238.0289, 'U'),
  'V': (1e-4/50.9415, 'V'),
  'W': (1e-4/183.84, 'W'),
  'Y': (1e-4/88.90585, 'Y'),
  'Yb': (1e-4/173.04, 'Yb'),
  'Zn': (1e-4/65.39, 'Zn'),
  'Zr': (1e-4/91.224, 'Zr'),
}

routes = [
    (r'/_plot_compositions', CompositionPlotHandler),
    (r'/([0-9]+)/compositions\.csv', CompositionPlotHandler),
]
