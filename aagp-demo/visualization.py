
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib.colors import ListedColormap#,LinearSegmentedColormap
from matplotlib import colors as mpl_colors

def SETSTYLE(style=['bmh','default','seaborn','fivethirtyeight'][0], clear = True):
    try:
        if clear:
            plt.style.use('default')
        plt.style.use(style)
        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams.update({'font.family': 'STIXGeneral'})
    except:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        if clear:
            plt.style.use('default')
        plt.style.use(style)

        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams.update({'font.family': 'STIXGeneral'})

    return

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)
def hex_to_rgb(hex):
    return tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def get_f38_palette(output_format = ['hex','rgb'][0]):
    F38_palette = ['#30a2da',
                '#fc4f30',
                '#e5ae38',
                '#6d904f',
                '#8b8b8b',
                '#e630e6',
                '#30309c',
                '#423030',
                '#30e67d',
                '#7c4ee6',
                '#a2305c',
                '#bde686',
                '#e68bb9',
                '#7ee6e5',
                '#8fe630'
                ]
    if 'h' in output_format:
        return F38_palette
    else:
        return np.asarray([hex_to_rgb(g) for g in F38_palette])
    


def get_global_colors(classic = False, maximize_visual_differencing=False, add_classic = False):
    '''
    # Get Global Colors
    This function grabs a bunch of colors and selects them such that they alternate between major color (regd, blue, green, yellow), with different shading.
    - If `maximize=True`, then all the colors will be selected in an order that maximizes their RGB distance.
    '''
    
    # [0] - instantiate the classic colors
    # ==========================================
    classic_colors = ['C%s'%(g) for g in range(0,32)] * int(add_classic)

    # [1] - grab the varieties of reds, blues, greens, and yellows
    # ==============================================================
    reds    = {
                0: 'red',                1: 'orangered',
                2: 'indianred',                3: 'darkred',
                4: 'lightcoral',                5: 'firebrick',
                6: 'tomato',                7: 'coral',
                }
    blues    = {
                0: 'blue',                1: 'navy',
                2: 'slateblue',                3: 'mediumslateblue',
                4: 'darkviolet',                5: 'magenta',
                6: 'purple',                7: 'royalblue',
                }
    greens    = {
                0: 'green',                1: 'lime',
                2: 'olivedrab',                3: 'springgreen',
                4: 'greenyellow',                5: 'aquamarine',
                6: 'palegreen',                7: 'seagreen',
                }
    yellows    = {
                0: 'gold',                1: 'orange',
                2: 'yellow',                3: 'navajowhite',
                4: 'goldenrod',                5: 'khaki',
                6: 'peru',                7: 'yellowgreen'
                }

    # [2] - grab the varieties of some "HARD" colors (4 groups of red, green, blue, yellow)
    # ===================================================
    hard_colors = [
        # group 1
        # ........
        'blue','gold','red','green',

        # group 2
        # ........
        'navy','peru','magenta','lime',

        # group 3
        # ........
        'deepskyblue','orange','deeppink','springgreen',

        # group 4
        # ........
        'aquamarine','yellow','sandybrown','olivedrab',
    ]

    # [3] - generate the coloring
    # ====================================================
    new_colors = []
    for i in range(len(list(reds.keys()))):
        colors      = [g[i] for g in [blues, yellows, reds, greens]]
        new_colors  += colors
    
    output_colors = new_colors + classic_colors if not classic else classic_colors + new_colors
    output_colors = hard_colors + classic_colors + new_colors if not classic else classic_colors + hard_colors + new_colors

    # [4] - if we want to maximize the visual differences between them, then we will apply maximin distancing.
    # =============================================================================================================
    if maximize_visual_differencing:
        from sklearn.metrics.pairwise import euclidean_distances

        rgb_codes = np.array([mpl_colors.to_rgb(g) for g in output_colors])*255
        x         = rgb_codes[0,:].reshape(1,-1)
        rgb_codes = np.delete(rgb_codes,0,axis=0)
        for i in range(rgb_codes.shape[0]):
            d = euclidean_distances(x,rgb_codes).min(0)
            idx = np.argmax(np.ravel(d))
            x   = np.vstack((x,rgb_codes[idx,:].reshape(1,-1)))
            rgb_codes = np.delete(rgb_codes,idx,axis=0)

        x = x.astype(int)
        output_colors = [rgb_to_hex(*x[i,:]) for i in range(x.shape[0])]
    return output_colors
        


def dawn_cmap(reverse = False):
    '''
    This function will return a custom colormap.
    '''
    # [1] - make the coloring
    # ==============================
    mats = np.matrix('[255 255 195;255 255 194;255 255 193;255 255 191;255 255 190;255 255 189;255 255 188;255 255 187;255 255 186;255 255 185;255 255 184;255 255 183;255 255 182;255 255 181;255 255 179;255 255 178;255 255 177;255 255 176;255 255 175;255 255 174;255 255 173;255 255 172;255 255 171;255 255 170;255 255 169;255 255 167;255 255 166;255 255 165;255 255 164;255 255 163;255 255 162;255 255 161;255 255 160;255 255 159;255 255 158;255 255 157;255 255 155;255 255 154;255 255 153;255 255 152;255 255 151;255 255 150;255 255 149;255 255 148;255 255 147;255 255 146;255 255 145;255 255 143;255 255 142;255 255 141;255 255 140;255 255 139;255 255 138;255 255 137;255 255 136;255 255 135;255 255 134;255 255 133;255 255 131;255 255 130;255 255 129;255 255 128;255 255 127;255 255 126;255 255 125;255 253 125;255 251 125;255 249 125;255 247 125;255 245 125;255 242 125;255 241 125;255 238 125;255 237 125;255 235 125;255 233 125;255 231 125;255 229 126;255 227 126;255 225 126;255 223 126;255 221 126;255 219 126;255 217 126;255 215 126;255 213 126;255 211 126;255 209 126;255 207 126;255 205 126;255 203 126;255 201 126;255 199 126;255 197 126;255 195 126;255 193 126;255 191 126;255 189 126;255 187 126;255 185 126;255 183 126;255 181 126;255 179 126;255 177 126;255 175 126;255 173 126;255 171 126;255 169 126;255 167 126;255 165 126;255 163 126;255 161 126;255 159 126;255 157 126;255 155 126;255 153 126;255 151 126;255 149 126;255 147 126;255 145 127;255 143 127;255 141 127;255 138 127;255 136 127;255 134 127;255 132 127;255 131 127;255 129 127;254 126 127;252 125 127;250 122 127;248 121 127;246 118 127;244 116 127;242 115 127;240 113 127;238 111 127;236 109 127;234 107 127;232 105 127;230 102 127;228 100 127;226 98 127;224 97 127;222 94 127;220 93 127;218 91 127;216 89 127;214 87 127;212 84 127;210 83 127;208 81 127;206 79 127;204 77 127;202 75 127;200 73 127;198 70 127;196 68 127;194 66 127;192 64 127;190 63 127;188 61 127;186 59 127;184 57 127;182 54 127;180 52 127;178 51 127;176 49 127;174 47 127;171 44 127;169 42 127;167 40 127;165 39 127;163 37 127;161 34 127;159 33 127;157 31 127;155 29 127;153 27 127;151 25 127;149 22 127;147 20 127;145 18 127;143 17 127;141 14 127;139 13 127;137 11 127;135 9 127;133 6 127;131 4 127;129 2 127;127 0 127;125 0 127;123 0 127;121 0 127;119 0 127;117 0 127;115 0 127;113 0 127;111 0 127;109 0 127;107 0 127;105 0 127;103 0 127;101 0 127;99 0 127;97 0 127;95 0 127;93 0 127;91 0 127;89 0 127;87 0 126;85 0 126;83 0 126;82 0 126;80 0 126;78 0 126;76 0 126;74 0 126;72 0 126;70 0 126;68 0 126;66 0 126;64 0 126;62 0 126;60 0 126;58 0 126;56 0 126;54 0 126;52 0 126;50 0 126;48 0 126;46 0 126;44 0 126;42 0 126;40 0 126;38 0 126;36 0 126;34 0 126;32 0 126;30 0 126;28 0 126;26 0 126;24 0 126;22 0 126;20 0 126;18 0 126;16 0 126;14 0 126;12 0 126;10 0 126;8 0 126;6 0 126;4 0 126;2 0 126;0 0 126]')
    mats = np.flipud(mats)

    # [2] - scale between 0-1
    # ==============================
    mats = mats/mats.max()
    if reverse:
        mats = np.flipud(mats)
    
    # [3] - generate the palette
    # ==============================
    cmap = ListedColormap(mats)

    # [4] - return
    # ==============================
    return cmap

def pretty_plot(ax, fig, gg1, gg2, gg3, projection=['2d','3d'], cmap=cm.jet, fill_2d_contour = True, add_2d_colorbar=False,):
    levSize1 = 100
    levSize2 = 10
    lev      = lambda g, levs: np.linspace(g.min(), g.max(), levs).tolist()


    if projection.lower() == '3d':
        alpha = 0.75
        ax.plot_surface(gg1,gg2,gg3,
                        rstride     = 1,
                        cstride     = 1,
                        cmap        = cmap,
                        alpha       = alpha,
                        antialiased = True,
                        shade       = True,
                        edgecolor   = 'none'
                        )
        ax.plot_wireframe(gg1,gg2,gg3,
                          rstride       = 1,
                          cstride       = 2,
                          linewidth     = 0.25,
                          color         = 'black',
                          alpha         = 0.125,
                          antialiased   = True
                          )
        try:
            ax.contour(gg1,gg2,gg3, cmap = cmap, alpha = 0.5, offset = gg3.min())
        except:
            pass
    else:
        if fill_2d_contour:
            # try:
                surface = ax.contourf(gg1,gg2,gg3,
                                    lev(gg3, levSize1),
                                    cmap = cmap,
                                    alpha= 0.75
                                    )
                ax.contour(gg1,gg2,gg3,
                        lev(gg3,levSize2),
                        cmap = cmap,
                        alpha= 1.0,
                        )
            # except:
            #     print('WARNING. No contour levels.')
        else:
            surface = ax.contourf(gg1,gg2,gg3,
                                  lev(gg3, levSize1),
                                  cmap = cmap,
                                  alpha = 0.375
                                  )
            ax.contour(gg1,gg2,gg3,
                       lev(gg3, levSize2), 
                       cmap = cmap,
                       linewidths = 1,
                       zorder = -1,
                       alpha = 1.0
                       )
        if add_2d_colorbar:
            divider = make_axes_locatable(ax)
            cax     = divider.append_axes('right',size = '5%', pad = 0.05)
            cb = fig.colorbar(surface, cax = cax, orientation = 'vertical')
            tick_locator = ticker.MaxNLocator(nbins=4)
            cb.locator   = tick_locator
            cb.update_ticks()
    return ax,fig



def fast_plot(ax, fig, bounds, func, projection=['2d','3d'][1],cmap = cm.jet, gsize=50, view_elevation = 15, view_rotation=330, square_axes=True, add_2d_colorbar=False):
    if type(bounds) != list:
        bounds = [bounds.min(0).tolist(), bounds.max(0).tolist()]
    
    # [1] - get the bounds and make the grid
    # ==========================================
    bLo,bHi = bounds
    g1,g2 = [np.linspace(bLo[g], bHi[g], gsize) for g in range(len(bLo))]
    gg1,gg2 = np.meshgrid(g1,g2)
    ggx     = np.vstack((np.ravel(gg1), np.ravel(gg2))).T
    ggy     = func(ggx)
    gg3     = np.asarray(ggy.reshape(-1,gsize))
    # [2] - plot the mesh    
    # ==========================================
    ax,fig = pretty_plot(ax, fig, gg1,gg2,gg3, projection = projection, cmap = cmap,add_2d_colorbar=add_2d_colorbar)
    if projection.lower() == '3d':
        ax.view_init(view_elevation, view_rotation)
    else:
        # plt.tight_layout()
        if square_axes:
            ax.set_aspect(1 / ax.get_data_ratio())
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel(r'$X_{1}$')
    ax.set_ylabel(r'$X_{2}$')
    try:
        ax.set_zlabel(r'$Y$')
        ax.set_zticklabels([])
    except:
        pass
    return ax,fig




def dawn_cmap(reverse = False):

    # reads in duskmap.txt and returns as a cmap.
    from matplotlib.colors import ListedColormap,LinearSegmentedColormap



    mats = np.matrix('[255 255 195;255 255 194;255 255 193;255 255 191;255 255 190;255 255 189;255 255 188;255 255 187;255 255 186;255 255 185;255 255 184;255 255 183;255 255 182;255 255 181;255 255 179;255 255 178;255 255 177;255 255 176;255 255 175;255 255 174;255 255 173;255 255 172;255 255 171;255 255 170;255 255 169;255 255 167;255 255 166;255 255 165;255 255 164;255 255 163;255 255 162;255 255 161;255 255 160;255 255 159;255 255 158;255 255 157;255 255 155;255 255 154;255 255 153;255 255 152;255 255 151;255 255 150;255 255 149;255 255 148;255 255 147;255 255 146;255 255 145;255 255 143;255 255 142;255 255 141;255 255 140;255 255 139;255 255 138;255 255 137;255 255 136;255 255 135;255 255 134;255 255 133;255 255 131;255 255 130;255 255 129;255 255 128;255 255 127;255 255 126;255 255 125;255 253 125;255 251 125;255 249 125;255 247 125;255 245 125;255 242 125;255 241 125;255 238 125;255 237 125;255 235 125;255 233 125;255 231 125;255 229 126;255 227 126;255 225 126;255 223 126;255 221 126;255 219 126;255 217 126;255 215 126;255 213 126;255 211 126;255 209 126;255 207 126;255 205 126;255 203 126;255 201 126;255 199 126;255 197 126;255 195 126;255 193 126;255 191 126;255 189 126;255 187 126;255 185 126;255 183 126;255 181 126;255 179 126;255 177 126;255 175 126;255 173 126;255 171 126;255 169 126;255 167 126;255 165 126;255 163 126;255 161 126;255 159 126;255 157 126;255 155 126;255 153 126;255 151 126;255 149 126;255 147 126;255 145 127;255 143 127;255 141 127;255 138 127;255 136 127;255 134 127;255 132 127;255 131 127;255 129 127;254 126 127;252 125 127;250 122 127;248 121 127;246 118 127;244 116 127;242 115 127;240 113 127;238 111 127;236 109 127;234 107 127;232 105 127;230 102 127;228 100 127;226 98 127;224 97 127;222 94 127;220 93 127;218 91 127;216 89 127;214 87 127;212 84 127;210 83 127;208 81 127;206 79 127;204 77 127;202 75 127;200 73 127;198 70 127;196 68 127;194 66 127;192 64 127;190 63 127;188 61 127;186 59 127;184 57 127;182 54 127;180 52 127;178 51 127;176 49 127;174 47 127;171 44 127;169 42 127;167 40 127;165 39 127;163 37 127;161 34 127;159 33 127;157 31 127;155 29 127;153 27 127;151 25 127;149 22 127;147 20 127;145 18 127;143 17 127;141 14 127;139 13 127;137 11 127;135 9 127;133 6 127;131 4 127;129 2 127;127 0 127;125 0 127;123 0 127;121 0 127;119 0 127;117 0 127;115 0 127;113 0 127;111 0 127;109 0 127;107 0 127;105 0 127;103 0 127;101 0 127;99 0 127;97 0 127;95 0 127;93 0 127;91 0 127;89 0 127;87 0 126;85 0 126;83 0 126;82 0 126;80 0 126;78 0 126;76 0 126;74 0 126;72 0 126;70 0 126;68 0 126;66 0 126;64 0 126;62 0 126;60 0 126;58 0 126;56 0 126;54 0 126;52 0 126;50 0 126;48 0 126;46 0 126;44 0 126;42 0 126;40 0 126;38 0 126;36 0 126;34 0 126;32 0 126;30 0 126;28 0 126;26 0 126;24 0 126;22 0 126;20 0 126;18 0 126;16 0 126;14 0 126;12 0 126;10 0 126;8 0 126;6 0 126;4 0 126;2 0 126;0 0 126]')
    mats = np.flipud(mats)
    mats = mats/mats.max()
    if reverse:
        mats = np.flipud(mats)
    cmap = ListedColormap(mats)

    return cmap