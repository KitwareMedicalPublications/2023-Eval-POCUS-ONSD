import seaborn as sb
import pandas as pd

PROBE_PRETTY = { 'butterfly-iq' : 'Butterfly iQ', 'clarius-l7hd' : 'Clarius L7HD', 'sonoque-l5c' : 'Sonoque L5C', 'sonivate' : 'Sonivate SonicEye', 'interson-spl01' : 'Interson SPL01' }
PROBE_ORDERED = [ 'Butterfly iQ', 'Clarius L7HD', 'Sonoque L5C', 'Sonivate SonicEye', 'Interson SPL01' ]
PROBE_PALETTE = { k : v for k, v in zip(PROBE_ORDERED, sb.color_palette('colorblind', n_colors=len(PROBE_ORDERED))) } 
CONTRAST_ORDERED = ['contrast_3cm -9.0', 'contrast_3cm -6.0', 'contrast_3cm -3.0', 'contrast_3cm 3.0', 'contrast_3cm 6.0', 'contrast_3cm h']
CONTRAST_PRETTY = {
    'contrast_3cm -9.0' : '-9 db', 
    'contrast_3cm -6.0' : '-6 db',
    'contrast_3cm -3.0' : '-3 db',
    'contrast_3cm 3.0' : '3 db',
    'contrast_3cm 6.0' : '6 db',
    'contrast_3cm h' : '>15 db'
}