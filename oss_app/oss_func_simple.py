"""
Created: Adapted from oss_functions created 01-18-2023 Kevin N. Schneider (KNS)
Last modified: 04-09-2024 KNS

Module script containing functions used to manipulate operant social self-administration (OSS) dataset.
This is an abridged version focused on basic wrangling and preprocessing to create a starting point
    for further analysis with the calculated social index scores.

Raw pandas dataframe is initialized into class DataDF, filtered based on input into
    a FiltDF object.
The scope of this script mostly ends with access to FiltDF's `df` and `df_scaled` properties,
    which contain the filtered dataframes with raw and scaled (z-scored) values, respectively,
    both with an extra columns containing the social index score per subject (row).

"""

# %% May need these
import random  # For si_plots
import pdb  # Keep for now, for debugging
import pdb  # for debugging, see pdb.set_trace()
from turtle import st
from typing import Union
import altair as alt
import marimo as mo

# %% Imports
# NOTE: many of these unused and so commented out in this simplified script,
#   but here in case needed
# data
import pandas as pd
# from pandas.api.types import is_string_dtype # Keep for now, might be used by FiltDF methods
# from pandas.api.types import is_numeric_dtype # Keep for now, might be used by FiltDF methods
import numpy as np  # Moved from later, standard practice

# plotting
import matplotlib.pyplot as plt
from matplotlib import colors  # Used by CmapBar if defined here
from matplotlib.colors import ListedColormap
# from matplotlib.patches import Patch, Rectangle # Keep for now
# from matplotlib import cm # Keep for now
from matplotlib.ticker import PercentFormatter  # Keep for now
from matplotlib.lines import Line2D  # For pca_biplot legend
import seaborn as sns
import pprint  # For pretty printing of dicts

# stats and clustering
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
# from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler  # Keep for now
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import calinski_harabasz_score, homogeneity_score, completeness_score, \\
#     normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, davies_bouldin_score
# from sklearn.neighbors import kneighbors_graph
# from sklearn.model_selection import cross_val_score
from scipy import stats
from scipy.stats import norm as normf
from scipy.stats import ks_2samp  # Keep for now
# from scipy import linalg

# misc
import os
from pathlib import Path
# from itertools import chain, cycle # Keep for now
import warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings('ignore')
# import numpy as np # Moved to top


# %% Data wrangling


def check_path(path: str, create_fold=False):
    """
    Checks
    For folders leave out the / at the end; eg, path/folder not path/folder/
    :param path: path to file or directory to check
    :param create_fold: bool: if True and path is directory, will create a new folder if it can't find it
    :return:
    """
    path = Path(rf"{path}").resolve()
    # check if path is folder or file, pretty brute force approach though
    # isdir = not os.path.splitext(path)[-1]
    isdir = path.is_dir()
    # check if folder exists and whether to make new one
    match (isdir, create_fold):
        case (True, False):
            raise Exception(f'folder not found:\n"{path}')
        case (False, True):
            print(f'folder not found, making new directory:\n"{path}"')
            # os.mkdir(path)
            path.mkdir(parents=True, exist_ok=True)
            return True
        case (False, _):
            raise Exception(
                f'file "{os.path.basename(path)}" not found in dir "{os.path.dirname(path)}"')
        case _:
            return True


def filter_df(df_raw, indices, id_var, excludev=[], fparams={}, print_df=False,):
    assert indices is not [''], 'index variables not found//'
    assert id_var != '', 'mouse id variable missing..'
    assert fparams is not {}, 'filter parameters not found'
    # does this function create a class or does the rawdf class loop this to create multiple df attributes?
    csv_indices = list()
    csv_filts = list()
    # pdb.set_trace()
    # extract included data and group by indices
    df_full = df_raw.loc[:, ~df_raw.columns.isin(excludev)].groupby(
        indices).first().reset_index()
    for idx, val in fparams.items():
        # filter by parameters
        df_full = df_full.loc[(df_full[idx].str.contains(val))]
        csv_indices.append(idx)
        csv_filts.append(val)
    # create name for filtered df
    csv_name = 'filtered'
    for idx, filt in zip(csv_indices, csv_filts):
        csv_name = csv_name + f'_[{idx}={filt}]'
    if print_df:
        df_full.to_csv(csv_name + '.csv')
    if fparams != {}:
        print(f'---using df with filters: "{csv_name}"')
    else:
        print(f'---no filters given.')
    return df_full, csv_name


def fix_names(strings: list[str] = None):
    """correct whitespaces or hyphens in column labels"""
    assert strings is not None, "no strings provided to fix_names"
    new_strings = [x.replace(" ", "_").replace("-", "_") for x in strings]
    return new_strings


# %% Plotting

def get_minmax(plotdata: np.ndarray) -> tuple:
    return (min(plotdata), max(plotdata))


def test_minmax(plotdata: np.ndarray, curr_mm=(0, 0), debug=False):
    curr_min, curr_max = curr_mm
    vmin, vmax = get_minmax(plotdata)
    mintest = 1 if vmin < curr_min else 0
    maxtest = 1 if vmax > curr_max else 0
    match (mintest, maxtest):
        case (1, 0):
            curr_min = vmin
        case (0, 1):
            curr_max = vmax
        case (1, 1):
            curr_min = vmin
            curr_max = vmax
        case _:
            curr_min, curr_max = vmin, vmax
    if debug:
        print(mintest, maxtest)
    return curr_min, curr_max


def round_to_nearest(base, *nums, round_up=True):
    """round num to nearest `base`,
    round up will round number away to higher magnitude, away from 0.
    """
    for num in nums:
        if (num < 0 and round_up) or (num > 0 and not round_up):
            # rounds negative number up, positive down
            bias = base//2
        else:
            # rounds negative number down, positive up
            bias = base
        roundt = num % base
        n = num + bias if roundt != 0 else num
        yield n - (n % base)


def roundit(*args):
    divn = 1  # default
    if len(*args) > 1:
        # iterable of values given
        out = []
        for arg in iter(*args):
            vmxtf = 1 if arg > 16 else 0
            match vmxtf:
                case 1 if arg >= 100:
                    divn = 20
                case 1 if arg >= 50:
                    divn = 10
                case 1:
                    divn = 5
                case _:
                    divn = 1
            out.append(int(np.rint(arg / divn) * divn))
        return out
    else:
        # single value given
        arg, *_ = args
        return int(np.rint(arg / divn) * divn)


def set_size(w, h, ax=None):
    """w, h: width, height in inches """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)

    # fig, ax = plt.subplots(figsize=(2, 2))
    # sns.histplot(x=distvar, data=df1, kde=False, multiple='stack', stat='percent', fill=True, binwidth=0.5, ax=ax)  # bins=[-5,-4,-3,-2,-1,0,1,2,3,4,5])
    # sns.histplot(x=distvar, data=df2, kde=False, multiple='stack', stat='percent', fill=True, binwidth=0.5, ax=ax)  # bins=[-5,-4,-3,-2,-1,0,1,2,3,4,5])
    # # p = [h.get_height() for h in ax.patches]
    # ax2 = ax.twinx()
    # sns.kdeplot(x=distvar, data=df1, fill=True, bw_adjust=1, color='blue', ax=ax2)  # stat='percent', kde=True, fill=True,bins=[-5,-4,-3,-2,-1,0,1,2,3,4,5])
    # sns.kdeplot(x=distvar, data=df2, fill=True, bw_adjust=1, color='purple',
    #             ax=ax2)  # stat='percent', kde=True, fill=True,bins=[-5,-4,-3,-2,-1,0,1,2,3,4,5])
    # ax.patches.clear()
    # ax2.get_yaxis().set_visible(False)
    # for axn in fig.axes:
    #     axn.spines.right.set_visible(False)
    #     axn.spines.top.set_visible(False)
    # ax.set_xlabel('Cum. Z-Score')
    # ax.set_ylabel('# Subjects (%)')
    # ax.set_xlim(-12, 12)
    # ax.set_xticks(np.linspace(-10, 10, 11))
    # ax.set_ylim(0, 15)
    # ax.set_yticks(np.linspace(0, 15, 4))
    # ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    # fig.suptitle('males:blue, females:purple')


def plot_xcorr(data: pd.DataFrame, hcmap: ListedColormap, xlabels=[], ylabels=[], xcorr=False, transp=False,
               var_order=None, var_order_labels=None, **hmap_kwargs):
    if var_order:
        data = data.iloc[:, var_order]
        # if var_order_labels:
        #     labels = [var_order_labels[i] for i in var_order]
        #     match (xlabels, ylabels):
        #         case (True, False):
        #             xlabels = labels
        #         case (False, True):
        #             ylabels = labels
        #         case _:
        #             xlabels = labels
        #             ylabels = labels
    if not ylabels:
        ylabels = [v for v in range(
            len(data.columns))] if not xlabels else False
    if not xlabels:
        xlabels = [r for r in range(len(data))] if not ylabels else False

    if xcorr:
        # Ensure data is DataFrame before .corr()
        data_corr = pd.DataFrame(data).corr()
    if transp:
        data_to_plot = data_corr.transpose() if xcorr else data.transpose()
    else:
        data_to_plot = data_corr if xcorr else data

    if not hmap_kwargs:  # Keep original check, it's more Pythonic
        hmap_kwargs = {
            'annot': True, 'fmt': '.2f', 'vmin': -1, 'vmax': 1, 'square': True,
            'cbar': False, 'linewidth': 1, 'linecolor': 'k',
            'yticklabels': ylabels, 'xticklabels': False, 'annot_kws': {'fontsize': 8},
            'cbar_kws': {'ticks': np.linspace(-1, 1, 3), "orientation": "horizontal"}
        }
    hm_fig = sns.heatmap(data_to_plot, cmap=hcmap, **hmap_kwargs)
    if xcorr:
        return hm_fig, data_corr  # Return original correlation matrix as well
    else:
        return hm_fig


# %% Analysis
class AgglomerativeClusteringWrapper(AgglomerativeClustering):
    def predict(self, X):
        return self.labels_.astype(int)


def do_pca(data: pd.DataFrame, n_comp=3):
    '''
    :param data: dataframe with scaled data
    :param n_comp: number of components to yield, .95 to find best number
    :return: pca obj, principalComponents (fit_transformed) and PC_evr
    '''
    pca = decomposition.PCA(n_components=n_comp)
    principal_components = pca.fit_transform(data)
    pc_evr = pca.explained_variance_ratio_
    return pca, principal_components, pc_evr


# %% All fns

def chk_debug(msg: str, debug=False):
    """
    Checks if debug == True for printing
    :param msg:
    :param debug:
    :return:
    """

    if debug:
        print(msg)  # Completed function


def make_cat(series: pd.Series):
    # make sure series data is integer or string
    if float in list(series.apply(type)):  # Check if any element is of type float
        # Ensure all float values are not whole numbers
        assert all(not x.is_integer() for x in series if isinstance(x, float)), \
            'can only use labels or integers for categories, floats must not be whole numbers'

    category_dict = {
        'codes': pd.Categorical(series).codes,
        'labels': pd.Categorical(series).categories}
    return category_dict


# Add quick_filter function
def quick_filter(dataf: pd.DataFrame, filters: dict = {}):
    """Simpler function to filter a dataframe `dataf` based on dictionary of `filters`.
    raises error if filter is empty
    :param dataf: dataframe to filter
    :type dataf: pd.DataFrame
    :param filters: dictionary of column:value items for dataframe, defaults to {}
    :type filters: dict, optional

    :return: returns filtered df.
    :rtype: pd.DataFrame
    """
    assert filters != {}, '--filters dictionary is empty.'
    query_string = ' & '.join(
        [f'`{var}`==\"{val}\"' if isinstance(val, str) else f'`{var}`=={val}'
         for var, val in filters.items()]
    )
    filtered_df = dataf.query(query_string)
    return filtered_df


# %% Classes
class PlotSetts:
    """Object containing settings for plotting.

    Settings:
        - *title*: label for plot, e.g., 'SI Score Distributions'
        - *xlims*: min and max for x-axis as (min, max)
        - *ylims*: min and max for y-axis as (min, max).
    """

    def __init__(self, data=None, title='', ylims: tuple[int, int] = (), xlims: tuple[int, int] = (), rndn=(), figsize=(), *args, **kwargs):
        # pdb.set_trace()
        arguments = locals()
        default_vals = {
            'ylims': (0, 12),
            'xlims': (-10, 10),
            'rndn': 1,
            # can add more metric-based defaults here if wanted
        }
        for arg, val in arguments.items():
            try:
                len(val) == 0
            except TypeError:
                continue
            if len(val) == 0:
                if arg in default_vals.keys():
                    tf = 1 if ((data is not None) and ('ylims' in arg)) else 0
                    match tf:
                        case 1:
                            self.__setattr__(arg, (min(data), max(data)))
                        case _:
                            self.__setattr__(arg, default_vals.get(arg))
            else:
                self.__setattr__(arg, val)
        self.title = title
        self.figsize = figsize if figsize != () else (2, 2)

    def set_setting(self, setting, value):
        match hasattr(self, setting):
            case True:
                self.__setattr__(setting, value)
                # self.chk_debug(f'{setting} changed to {value}', self.debug)
            case False:
                self.__setattr__(setting, value)
                # self.chk_debug(f'{setting} not already present, new attribute made with value: {value}', self.debug)

    def map_settings(self, dict_map: dict):

        missed_keys = []
        for key, val in dict_map.items():
            try:
                self.set_setting(key, val)
            except KeyError:
                # print(f'could not set attribute: {key}: {val}')
                missed_keys.append(key)
        if missed_keys:
            pass
            # self.chk_debug(f'keys missed:  {missed_keys}', self.debug)

    def check_settings(self):
        """Prints current settings."""
        pp = pprint.PrettyPrinter(indent=3, depth=3)
        pp.pprint(vars(self))


class DataDF:
    """
    Takes in csv given along with variable parameters to yield object with raw and processed dataframes.

    Contains methods for plotting that take in further parameters to make plots seen in the paper.
    The following plots are included:
    -

    Params:
    -------
    csv_data :

    """

    def __init__(self, df: pd.DataFrame | None = None, properties: dict = {}, debug=False):
        self.csv_path = None
        if df is not None:
            assert isinstance(
                df, pd.DataFrame), 'df must be a pandas DataFrame'
            self.raw_df = df
        else:
            self.raw_df = pd.DataFrame()  # need to run get_rawdf to load data from csv otherwise
        self.debug = debug
        self.properties = properties
        # set parameters
        self.id_variable = properties['subject_id_variable']
        self.grouping_variable = properties['grouping_variable']
        self.sex_variable = properties['sex_variable']
        self.other_indices = properties['index_variables']
        self.indices = self.id_variable + self.sex_variable + self.grouping_variable + \
            self.other_indices
        properties['indices'] = self.indices
        self.metrics_included = properties['metric_variables']
        properties['metrics_included'] = self.metrics_included

    def c_debug(self, msg):
        """
        Class implementation of check debug for printing
        :param msg:
        :param debug:
        :return:
        """
        chk_debug(msg, self.debug)

    def get_rawdf(self, csv_path):
        if self.raw_df is not None and not self.raw_df.empty:
            self.c_debug('debug: raw df already loaded, skipping...')
            return
        assert isinstance(
            csv_path, str), f'incorrect format for path to file given'
        assert csv_path.endswith(
            '.csv'), f'"{os.path.basename(csv_path)}" is not a csv file'
        assert os.path.exists(csv_path), f'file not found: "{csv_path}"'
        self.raw_df = pd.read_csv(csv_path)
        self.csv_path = csv_path
        self.c_debug('debug: loaded csv...')

        # filter out columns
        to_include = self.indices + self.metrics_included
        filter_list = [c for c in self.raw_df.columns if c not in to_include]
        self.raw_df.drop(columns=filter_list, inplace=True, errors='ignore')

        # check properties available among variables
        self.vars = self.raw_df.columns.values
        # self.c_debug('debug: vars...')
        # if self.debug:
        # print(f'{self.csv_path}')
        # self.c_debug(f'debug: print...')
        # pp = pprint.PrettyPrinter(indent=3, depth=3)
        # pp.pprint(self.raw_df)

    def fix_columns(self, columns: list[str] = None):
        """correct whitespaces or hyphens in column labels"""
        if columns is None:
            columns = self.raw_df.columns.values
        new_columns = [x.replace(' ', '_').replace('-', '_') for x in columns]
        mapping = {c: nc for c, nc in zip(columns, new_columns)}
        self.raw_df.rename(columns=mapping, inplace=True)
        fixed_raw_df = self.raw_df.dropna(axis=1, inplace=False)  # drop empty columns
        self.raw_df = fixed_raw_df

    def define_groups(self, group_variables: list[str], label_sep='-', column_name='Group'):
        """Define a new column with subject's group as determined by combination of
        given list of variables (columns). New labels are made in the order the variables
        appear in the list, and separated by character defined in `label_sep`.

        :param group_variables: list of grouping variables to combine
        :type group_variables: list[str]
        :param label_sep: string separator for new group labels, defaults to '-'
        :type label_sep: str, optional
        :param column_name: new column name for group variable, defaults to 'Group'
        :type column_name: str, optional
        """

        assert group_variables != [], 'empty variables list provided'
        self.unique_group = column_name
        if column_name not in self.raw_df.columns:
            self.properties['indices'].append(column_name)
            self.raw_df.insert(
                1, column_name, self.raw_df[group_variables].agg(label_sep.join, axis=1))
        else:
            self.c_debug(
                f'column "{column_name}" already exists, not redefining it.')

    def average_columns(self, columns: list[str], col_name: str = None):
        """create new column labeled `col_name` with the average across variables in `columns` for each subject."""
        if col_name is None:
            col_name = '.'.join(columns) + '_mean'
        self.raw_df.insert(
            2, col_name, self.raw_df[columns].agg(np.mean, axis=1))

    def get_diffs(self, label, var1, var2):
        """Creates a difference column between variable 1 and 2, eg. (post-stress - pre-stress) average rewards

        :param label: label for new column created
        :type label: str
        :param var1: column 1 name
        :type var1: str
        :param var2: column 2 name
        :type var2: str
        """
        # self.raw_df[label] = [np.divide(x, y, out=np.zeros_like(y), where=y!=0).flatten()[0] for x, y in zip(self.raw_df[var2].values, self.raw_df[var1].values)]
        self.raw_df[label] = [np.subtract(x, y) for x, y in zip(
            self.raw_df[var2].values, self.raw_df[var1].values)]

    # indices=[''], id_var='', excl_vars=[''], filt_params={}):
    # TODO: remove expected filters -- filtering now done before this
    def get_filt_df(self, filtparams: dict, excl_vars_new=''):
        """filter dataframe based on `filtparams`

        :param filt_params: _description_, defaults to {}
        :type filt_params: dict, optional
        :param excl_vars: list of column names to filter out, defaults to ['']
        :type excl_vars: list, optional
        :param id_var: column variable containing ID labels for subjects
        :type id_var: str

        :return: returns filtered df as a FiltDF object.
        :rtype: FiltDF class
        """
        if filtparams == {}:
            self.c_debug('No filt params, including all animals...')
        indices = self.properties['indices']
        id_var = self.properties['subject_id_variable']
        # excl_vars = excl_vars_new if excl_vars_new else self.properties['excluded_vars']
        filt_df, csv_name = filter_df(
            self.raw_df, indices, id_var, excl_vars_new, filtparams)
        self.c_debug(f'made: {csv_name}')
        self.filt_df = FiltDF(filt_df, csv_name, self.properties, self.debug)
        return self.filt_df


class FiltDF(DataDF):

    def __init__(self, filt_df, csv_name, properties, debug, *args):
        self.csv_name = csv_name
        self.df = filt_df
        super().__init__(filt_df, properties, debug)  # borrow from DataDF

        # metrics and data saved separately for processing
        self.df_idx = self.df.loc[:, self.df.columns.isin(
            properties['indices'])]  # just the index columns
        self.df_vars = self.df.loc[:,
                                   # just the metric columns, no grouping indices
                                   ~self.df.columns.isin(properties['indices'])]
        self.data_array = self.df_vars.values  # just the data ie values for df_metrics

        self.id_var = properties['subject_id_variable']
        # column values for id variable
        self.id_vals = list(self.df[properties['subject_id_variable']].values)
        self.group_col = self.df[properties['grouping_variable']]
        self.group_vals = list(self.group_col.values)
        self.group_categories = {
            'codes': pd.Categorical(self.group_col).codes,
            'labels': pd.Categorical(self.group_col).categories}
        self.idx_vars = list(self.df_idx.columns.values)
        self.metric_vars = list(self.df_vars.columns.values)
        self.stats = pd.DataFrame(columns=['test_name', 'value', 'p_val'])
        self.c_debug('Filter success!')

    def scale_data(self, scaletype=2):
        # TODO: per group
        match scaletype:
            case 0:
                stype = 'minmax (0-1)'
                data = MinMaxScaler().fit_transform(self.data_array)
            case 2:
                stype = 'robust scaler (Z)'
                data = StandardScaler().fit_transform(self.data_array)
            case _:
                stype = 'standard (Z)'
                data = RobustScaler().fit_transform(self.data_array)
        self.data_scaled = data

        self.df_scaled = pd.concat([self.df_idx, pd.DataFrame(columns=self.df_vars.columns, index=self.df_idx.index, data=data)],
                                   axis=1)
        if self.debug:
            print(f'...performed {stype} scaling of data:\n', data)

    def get_si(self, exclude=[], label=None):
        """
        Calculate cumulative score from z-scored metrics in data for animals in dataframe provided.
        """

        if not label:
            attr_label = 'si_scores'
            label = 'SI_scores'
        else:
            attr_label = label
        scaled_X = self.data_scaled[:, [
            i not in exclude for i in self.metric_vars]]
        self.__setattr__(attr_label, np.array(
            [sum(scaled_X[n, :]) for n in range(len(scaled_X))], dtype='float'))
        for d in (self.df, self.df_vars, self.df_scaled):
            d[label] = self.__getattribute__(attr_label)
        # return self.df[[self.properties['subject_id_variable'], 'SI_scores']]

    def clean_df(self, return_id_col: str = None):
        """get rid of subject rows that are missing values.
        if `return_id_col` set to column, prints the values that were removed for given column.
        """
        nan_rows = self.df_scaled.isna().any(axis=1)
        nonnan_rows = self.df_scaled.notna().all(axis=1)
        if return_id_col is not None:
            nanmice = self.df.loc[nan_rows, return_id_col].values
            print('mice ids with missing values:', nanmice, sep='\n')
        self.df_scaled = self.df_scaled.dropna()
        self.df = self.df.loc[nonnan_rows, :]

    def map_colors(self, data_array: np.ndarray = [], color_set=['viridis_r', 'plasma_r'], cmap_rng=None):

        if not data_array:
            data_array = self.si_scores

        self.cset = color_set
        num_sex = len(np.unique(self.df['Sex'].values))

        if not cmap_rng:
            cmap_rng = (.4, 1)

        match num_sex:
            case 2:  # 2 sexes present
                # default colors for male and female animals
                plot_colors = []
                self.c_debug(
                    f'using {self.cset} as color sets for 2 groups (default:sexes)')
                for i, cset in enumerate(self.cset):
                    plot_colors.append(
                        CmapSet(cset, data_array, cmap_range=cmap_rng))
            # only males present
            case 1 if 'Male'.casefold() in self.df['Sex'].values:
                plot_colors = CmapSet(
                    self.cset[0], data_array, cmap_range=cmap_rng)
                self.c_debug(f'using {self.cset[0]} as color set for males')
            # only females present
            case 1 if 'Female'.casefold() in self.df['Sex'].values:
                plot_colors = CmapSet(
                    self.cset[1], data_array, cmap_range=cmap_rng)
                self.c_debug(f'using {self.cset[1]} as color set for females')
            case _:  # default to plasma cmap in any other case
                plot_colors = CmapSet(
                    self.cset[0], data_array, cmap_range=cmap_rng)
                self.c_debug(f'using {self.cset[1]} as default color set')

        self.plot_colors = plot_colors

    def _split_data_by_group(self, compvar: str, groupvar: str, filters: dict) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Splits filtered dataframe into two arrays based on a grouping variable.
        """
        df_to_use = self.df_scaled
        if filters:
            query_str = ' & '.join(
                [f'`{var}`=="{val}"' if isinstance(
                    val, str) else f'`{var}`=={val}' for var, val in filters.items()]
            )
            df_to_use = df_to_use.query(query_str)

        comp_grps = df_to_use.groupby(by=groupvar)

        if len(comp_grps) != 2:
            raise ValueError(
                f"Expected 2 groups after filtering, but found {len(comp_grps)}. Please filter data to two groups."
            )

        comp_arr_labels = list(comp_grps.groups.keys())
        comp_arrays = comp_grps[compvar].apply(
            lambda x: x.to_numpy() if isinstance(x, pd.Series) else np.array(x)
        ).values

        return comp_arrays[0], comp_arrays[1], comp_arr_labels

    def _calculate_distribution(self, data_array: np.ndarray, f_axis: np.ndarray) -> tuple[np.ndarray, dict]:
        # Fit a Gaussian Mixture Model (MLE for single Gaussian)
        f = np.ravel(data_array).astype(np.float64).reshape(-1, 1)
        g = mixture.GaussianMixture(
            n_components=1, covariance_type='full', random_state=0)
        g.fit(f)
        weights, means, covars = g.weights_, g.means_, g.covariances_

        mean = means[0, 0]
        std_dev = np.sqrt(covars[0, 0, 0])

        pdf = normf.pdf(f_axis, loc=mean, scale=std_dev).ravel()

        # This will make the curve integrate to 100 over its range
        dist_freq = pdf * 100  # scale for percentage density

        gmm_params = {'mean': mean, 'std': std_dev}
        return dist_freq, gmm_params

    def _calculate_intersection_and_overlap_alt(self, gmm_params1: dict, gmm_params2: dict, rangex: list) -> tuple[float, float]:
        """
        Calculates the intersection point and overlap coefficient of two normal distributions.
        """
        def solve(m1_s, m2_s, std1_s, std2_s):
            a = 1 / (2 * std1_s ** 2) - 1 / (2 * std2_s ** 2)
            b = m2_s / (std2_s ** 2) - m1_s / (std1_s ** 2)
            c = m1_s ** 2 / (2 * std1_s ** 2) - m2_s ** 2 / \
                (2 * std2_s ** 2) - np.log(std2_s / std1_s)
            roots = np.roots([a, b, c])
            valid_roots = [r for r in roots if rangex[0] < r < rangex[1]]
            if not valid_roots:
                return roots[np.argmin(np.abs(roots - np.mean([m1_s, m2_s])))] if len(roots) > 0 else np.mean([m1_s, m2_s])
            return valid_roots[0] if len(valid_roots) == 1 else valid_roots[np.argmin(np.abs(valid_roots - np.mean([m1_s, m2_s])))]

        m1, s1 = gmm_params1['mean'], gmm_params1['std']
        m2, s2 = gmm_params2['mean'], gmm_params2['std']

        intersect_x = solve(m1, m2, s1, s2)

        if abs(m1) > abs(m2):
            area = normf.cdf(intersect_x, m1, s1) + \
                (1. - normf.cdf(intersect_x, m2, s2))
        else:
            area = normf.cdf(intersect_x, m2, s2) + \
                (1. - normf.cdf(intersect_x, m1, s1))

        return intersect_x, area

    def _calculate_intersection_and_overlap(
        self,
        gmm_params1: dict, gmm_params2: dict,
        x_range: list | np.ndarray
    ):
        """
        Calculate overlap coefficient (using the GMM-derived parameters)
        Uses the raw norm.pdf for the coefficient, then multiplies by 100 for percentage
        """
        # using estimated loc and scale from GMM parameters
        mean1_gmm_est, std1_gmm_est = gmm_params1['mean'], gmm_params1['std']
        mean2_gmm_est, std2_gmm_est = gmm_params2['mean'], gmm_params2['std']

        pdf_group1_normalized_for_overlap = normf.pdf(
            x_range, loc=mean1_gmm_est, scale=std1_gmm_est)
        pdf_group2_normalized_for_overlap = normf.pdf(
            x_range, loc=mean2_gmm_est, scale=std2_gmm_est)

        min_of_normalized_pdfs = np.minimum(
            pdf_group1_normalized_for_overlap, pdf_group2_normalized_for_overlap)
        dx_plot = x_range[1] - x_range[0]
        # This is the [0,1] coefficient
        overlap_coefficient = np.sum(min_of_normalized_pdfs) * dx_plot
        overlap_percentage_value = overlap_coefficient * 100

        print(f"--Calculated Overlap Coefficient: {overlap_coefficient:.4f}")
        print(
            f"--Calculated Overlap Percentage: {overlap_percentage_value:.2f}%")
        return overlap_coefficient, overlap_percentage_value

    def _perform_ks_test(self, data_array1: np.ndarray, data_array2: np.ndarray) -> tuple[float, float]:
        """
        Performs and reports the Kolmogorov-Smirnov test between two groups.
        """
        ks_stat, ks_pval = ks_2samp(
            data_array1.reshape(-1), data_array2.reshape(-1))
        print(
            f'----Kolmogorov-Smirnov test between groups: {ks_stat:.2f},  p-val: {ks_pval:.5f}')
        return ks_stat, ks_pval

    def compare_dists_altair(self, compare_metric='', group_variable='', filters={}, max_y=40,
                             rangex: list | None = [-2.5, 2.5], bin_width=1, dot=False,
                             user_colors=None, user_labels: list[str, str] = None,
                             set_size_params=None, legend=False, hide_text=False,
                             alpha1=0.7, alpha2=0.4):
        """
        Compares distribution between two groups for a given variable using Altair.
        This function is a wrapper that utilizes helper methods to perform its core tasks:
        1. _split_data_by_group: Splits data into two arrays for comparison.
        2. _calculate_distribution: Calculates the GMM distribution for each group.
        3. _calculate_intersection_and_overlap: Finds the intersection and overlap coefficient.
        4. _perform_ks_test: Runs a Kolmogorov-Smirnov test.
        The final output is an Altair chart object.

        :param compvar: Comparison variable name.
        :type compvar: str
        :param groupvar: Grouping variable name.
        :type groupvar: str
        :param filters: Dictionary of filters to apply to the dataframe.
        :type filters: dict
        :param max_y: Maximum y-axis value.
        :type max_y: int
        :param rangex: Range for the x-axis, e.g., [-2.5, 2.5].
        :type rangex: list
        :param binw: Bin width for density calculation scaling.
        :type binw: int
        :param dot: Whether to plot the intersection point.
        :type dot: bool
        :param user_colors: Tuple of two colors for the groups.
        :type user_colors: tuple[str, str], optional
        :param user_labels: List of two labels for the groups.
        :type user_labels: list[str, str], optional
        :param set_size_params: Tuple (width, height) in inches for the plot size.
        :type set_size_params: tuple[float, float], optional
        :param legend: Whether to display the legend.
        :type legend: bool
        :param hide_text: Whether to hide all text (titles, labels, ticks).
        :type hide_text: bool
        :param alpha1: Opacity for the first group's area.
        :type alpha1: float
        :param alpha2: Opacity for the second group's area.
        :type alpha2: float
        :return: An Altair chart object.
        :rtype: alt.Chart
        """
        assert compare_metric != '', 'no comparison variable for "compare_metric" chosen'
        if group_variable == '':
            group_variable = self.group_col if hasattr(self, 'group_col') and isinstance(
                self.group_col, str) else self.properties['grouping_variable']

        # 1. Split data into two groups
        array1, array2, comp_arr_labels = self._split_data_by_group(
            compare_metric, group_variable, filters)

        plot_title = f'{compare_metric=},  {group_variable=}'
        print(f'\n{plot_title}')

        # 2. Calculate distributions for each group
        concat_min = np.concatenate((array1, array2)).min()
        concat_max = np.concatenate((array1, array2)).max()
        if not rangex:  # If rangex is not provided, set it based on data but wider
            rangex = [round(concat_min-5), round(concat_max+5)]
        range_min, range_max = rangex
        smooth_factor = 500
        x_plot_range = np.linspace(
            # Increase smooth_factor for smoother curves
            range_min, range_max, smooth_factor)

        # Calculate distributions and get GMM parameters
        dist_freq_group1, gmm_params_group1 = self._calculate_distribution(
            data_array=array1, f_axis=x_plot_range,
        )
        mean1_gmm_est = gmm_params_group1['mean']
        std1_gmm_est = gmm_params_group1['std']

        dist_freq_group2, gmm_params_group2 = self._calculate_distribution(
            data_array=array2, f_axis=x_plot_range,
        )
        mean2_gmm_est = gmm_params_group2['mean']
        std2_gmm_est = gmm_params_group2['std']

        # Set user colors and labels
        _user_colors = user_colors if user_colors else ('#e45756', '#4c78a8')
        _lgnd_labels = user_labels if user_labels is not None else [
            str(lbl) for lbl in comp_arr_labels]

        print(
            f"--GMM Group 1: {_lgnd_labels[0]} Estimates: Mean={mean1_gmm_est:.2f}, Std={std1_gmm_est:.2f}")
        print(
            f"--GMM Group 2: {_lgnd_labels[1]} Estimates: Mean={mean2_gmm_est:.2f}, Std={std2_gmm_est:.2f}")

        # 3. Calculate intersection and overlap coefficient
        overlap_coeff, overlap_pct = self._calculate_intersection_and_overlap(
            gmm_params_group1, gmm_params_group2, x_plot_range)

        # 4. Perform and report Kolmogorov-Smirnov test
        ks_stat, ks_pval = self._perform_ks_test(array1, array2)

        # --- Altair Charting ---
        # Prepare data for Altair DataFrame

        data_dist_list = []
        for val_x, val_y in zip(x_plot_range, dist_freq_group1):
            data_dist_list.append(
                {'f_axis': val_x, 'dist_val': val_y, 'group': _lgnd_labels[0], 'alpha_val': alpha1})
        for val_x, val_y in zip(x_plot_range, dist_freq_group2):
            data_dist_list.append(
                {'f_axis': val_x, 'dist_val': val_y, 'group': _lgnd_labels[1], 'alpha_val': alpha2})
        source_dist = pd.DataFrame(data_dist_list)

        # Base chart properties
        plot_width = set_size_params[0] * 96 if set_size_params else 192
        plot_height = set_size_params[1] * 96 if set_size_params else 192
        if not max_y:
            concat_max = max(
                dist_freq_group1.max(), dist_freq_group2.max())
            max_y = round(concat_max + 10)
        # Create Altair chart
        area_chart = alt.Chart(source_dist).mark_area(
            line={'color': 'black', 'strokeWidth': 1.5, 'strokeOpacity': 1}, strokeWidth=0,
        ).encode(
            x=alt.X('f_axis:Q', title=f'Var: {compare_metric}', scale=alt.Scale(
                domain=rangex, nice=False, zero=False)),
            y=alt.Y('dist_val:Q', title='% Subjects', stack=False, scale=alt.Scale(
                domain=[0, max_y], nice=False, zero=True), axis=alt.Axis(format=".0f")),
            fill=alt.Fill(
                'group:N',
                scale=alt.Scale(domain=_lgnd_labels, range=_user_colors),
                legend=alt.Legend(title=group_variable) if legend else None,
            ),
            fillOpacity=alt.FillOpacity('alpha_val:Q', legend=None),
        )

        # chart = area_chart + line1_chart + line2_chart
        chart = area_chart
        # chart = line1_chart + line2_chart

        # Intersection point
        # if dot:
        #     source_intersect = pd.DataFrame(
        #         {'rx': [intersect_x], 'ry': [intersect_y_g1]})
        #     intersect_plot = alt.Chart(source_intersect).mark_point(
        #         color='black', size=60, filled=True).encode(x='rx:Q', y='ry:Q')
        # chart += intersect_plot

        # Titles and styling
        chart = chart.properties(
            width=plot_width,
            height=plot_height,
            title=alt.TitleParams(
                text=f'{plot_title}',
                subtitle=f'Overlap coeff: {overlap_coeff:.3f},  pct: {overlap_pct:.2f}%  |  K-S test: {ks_stat:.2f}, p-val: {ks_pval:.3f}',
                fontSize=12, subtitleFontSize=10, subtitleFontStyle='italic', anchor='middle', offset=10
            )
        )  # .configure_view(strokeWidth=0)

        if hide_text:
            chart = chart.configure_axis(labels=False, title=None, ticks=False, grid=False).properties(
                title=alt.TitleParams(text="", subtitle=""))
        else:
            # .encode(y=alt.Y('dist_val:Q', title='% Subjects', stack=False, scale=alt.Scale(
            chart = chart
            # domain=[0, max_y], nice=False, zero=False), axis=alt.Axis(format=".0f")))

        return chart

    def compare_dists(self, compvar='', groupvar='', filters={}, max_y=40,
                      rangex=[-2.5, 2.5], binw=1, kde=True, hist=True, dot=False,
                      user_colors=None, user_labels: list[str, str] = None,
                      set_size_params=None, legend=False, hide_text=False,
                      alpha1=0.7, alpha2=0.4):
        """Compares distribution between two groups for given variable `compvar`, optionally filtered by `filters` dictionary of variable values
        --Currently not flexible for more than 2 groups, dataframe should be filtered down to two unique groups, via `filters`
        or beforehand, before running this.

        :param compvar: _description_, defaults to ''
        :type compvar: str, optional
        :param groupvar: _description_, defaults to ''
        :type groupvar: str, optional
        :param filters: _description_, defaults to {}
        :type filters: dict, optional
        :param max_y: _description_, defaults to 40
        :type max_y: int, optional
        :param rangex: _description_, defaults to [-2.5, 2.5]
        :type rangex: list, optional
        :param binw: _description_, defaults to 1
        :type binw: int, optional
        :param dot: _description_, defaults to False
        :type dot: bool, optional
        :param user_colors: _description_, defaults to None
        :type user_colors: _type_, optional
        :param user_labels: _description_, defaults to None
        :type user_labels: list[str, str], optional
        :param set_size_params: _description_, defaults to None
        :type set_size_params: _type_, optional
        :param legend: _description_, defaults to False
        :type legend: bool, optional
        :param hide_text: _description_, defaults to False
        :type hide_text: bool, optional
        :param alphas: _description_, defaults to None
        :type alphas: _type_, optional
        :param debug: _description_, defaults to False
        :type debug: bool, optional
        :return: _description_
        :rtype: _type_
        """
        assert compvar != '', 'no comparison variable for "compvar" chosen'
        if groupvar == '':
            groupvar = self.group_col

        if filters != {}:
            newdf = self.df_scaled.query(' & '.join(
                [f'{var}=="{val}"' for var, val in filters.items()]
            )
            )
            comp_grps = newdf.groupby(by=groupvar)
        else:
            comp_grps = self.df_scaled.groupby(by=groupvar)
        comp_arr_labels = list(comp_grps.groups.keys())
        comp_arrays = comp_grps[compvar].apply(np.array).values
        # pdb.set_trace()
        plot_title = f'{compvar} by {groupvar}'

        smooth_factor = 100
        f_axis = np.linspace(rangex[0], rangex[1], smooth_factor)
        # f_prob_axis = np.arange(rangex[0], rangex[1], 1)
        # concatenated data for fitting 2 components
        xtot = np.concatenate((comp_arrays[0], comp_arrays[1]))

        def get_dist(data, faxis, ncom=1):
            f = np.ravel(data).astype(np.float64)
            f = f.reshape(-1, 1)
            g = mixture.GaussianMixture(
                n_components=ncom, covariance_type='full')
            g.fit(f)
            weights = g.weights_
            means = g.means_
            covars = g.covariances_
            return f.ravel(), weights, means, covars, g

        def pdf2freq(data, pdist, xtot=None, bwin=1, smooth_factor=100):
            if xtot is None:
                xtot = len(data)
            self.c_debug(f'--DEBUG: % total based on: N = {xtot} subjects')
            fpy = pdist * (len(data) * bwin)
            # fpy = pdist * (xtot * bwin)
            fperc = (fpy / xtot) * smooth_factor
            return fperc

        f1, w1, m1, c1, g1 = get_dist(comp_arrays[0], f_axis, 1)
        py = w1 * normf.pdf(f_axis, loc=m1, scale=np.sqrt(c1)).ravel()
        # x1dist = pdf2freq(comp_arrays[0], py, len(xtot), binw)
        x1dist = pdf2freq(comp_arrays[0], py, None, binw)

        f2, w2, m2, c2, _ = get_dist(comp_arrays[1], f_axis, 1)
        py = w2 * normf.pdf(f_axis, loc=m2, scale=np.sqrt(c2)).ravel()
        # x2dist = pdf2freq(comp_arrays[1], py, len(xtot), binw)
        x2dist = pdf2freq(comp_arrays[1], py, None, binw)

        # ftot, wtot, mtot, ctot, g = get_dist(xtot, f_axis, 2)

        def solve(m1, m2, std1, std2):
            # pdb.set_trace()
            a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
            b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
            c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / \
                (2 * std2 ** 2) - np.log(std2 / std1)
            # print(a,b,c)
            return np.roots([a, b, c])

        m1 = m1.ravel()[0]
        m2 = m2.ravel()[0]
        s1 = np.sqrt(c1).ravel()[0]
        s2 = np.sqrt(c2).ravel()[0]

        # Get point of intersect
        result = solve(m1, m2, s1, s2)
        rx = result[1]
        ry = result[0]
        r1 = normf.pdf(rx, m1, s1)
        r1 = r1 * len(comp_arrays[0]) * binw
        r1 = r1 / len(xtot) * 100

        # plot dists
        plt.figure(figsize=(2, 2))
        if not user_colors:
            user_colors = ('red', 'blue')
        if not (alpha1 and alpha2):
            alpha1, alpha2 = (0.5, 0.5)

        # sns.histplot(x=ftot.ravel(), bins=8, binrange=[-2,2], kde=False, stat='percent', binwidth=0.5)
        plt.plot(f_axis, x1dist, c='k', lw=1)
        plt.plot(f_axis, x2dist, c='k', lw=1)
        if dot is True:
            plt.plot(rx, r1, 'o', c='k')

        # Plots integrated area
        # r = result[1]
        # lgnd_labels = np.unique(self.df[groupvar])
        lgnd_labels = np.unique(
            comp_arr_labels) if user_labels is None else user_labels
        olap1 = plt.fill_between(
            f_axis, 0, x1dist, alpha=alpha1, color=user_colors[0], lw=0, label=lgnd_labels[0])
        olap2 = plt.fill_between(
            f_axis, 0, x2dist, alpha=alpha2, color=user_colors[1], lw=0, label=lgnd_labels[1])
        # calculate kolmogorov-smirnov t
        ks_stat, ks_pval = ks_2samp(
            comp_arrays[0].reshape(-1), comp_arrays[1].reshape(-1))
        print(
            f'\t--Kolmogorov-Smirnov test between groups: {ks_stat:.2},  p-val: {ks_pval:.5f}')

        # integrate overlap
        if abs(m1) > abs(m2):
            area = normf.cdf(rx, m1, s1) + (1. - normf.cdf(rx, m2, s2))
        else:
            area = normf.cdf(rx, m2, s2) + (1. - normf.cdf(rx, m1, s1))
        print(f"\n{plot_title} - Area under curves: {area:.3f}")
        plt.xlabel(f'Var: {compvar}')
        # plt.title(f'{plot_title}\nOverlap coeff: {area:.3f}\nK-S test: ')
        plt.ylabel('% Subjects')
        ax = plt.gca()
        if set_size_params:
            set_size(set_size_params[0], set_size_params[1], ax=ax)
        else:
            ax.set_aspect(abs((rangex[1] - rangex[0]) / (0 - max_y)))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlim(rangex[0], rangex[1])
        # ax.set_xticks(np.linspace(rangex[0], rangex[1], 3))
        ax.set_ylim(0, max_y)
        # ax.set_yticks(np.linspace(0, max_y, 6))
        plt.title(f'Overlap coeff: {area:.3f}\nK-S test: {ks_stat:.2},  p-val: {ks_pval:.3f}',
                  fontsize=10, fontstyle='italic')
        plt.suptitle(f'{plot_title}')
        plt.subplots_adjust(top=0.85)

        if not hide_text:
            # lgnd_labels = np.unique(self.df[groupvar])
            # plt.legend(legend, labels=lgnd_labels)

            ax.xaxis.set_tick_params(labelbottom=True)
        if legend:
            ax.legend(handles=[olap1, olap2], loc='best', fontsize=8)
        if hide_text:
            # Hide X and Y axes label marks
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            plt.title(None)
            plt.suptitle(None)
            plt.xlabel(None)
            plt.ylabel(None)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

    def dists_by_groupvar(self, data: pd.DataFrame = None, label: str = None, var_list: list = None,
                          groupvar: str = None, filters: dict = {}, groupvar_labels: list[str, str] = None,
                          rangex=[-2.5, 2.5], max_y=50, binw=1, set_size_params: tuple = None,
                          hide_text: bool = False, legend: bool = False, user_colors: list[str, str] = None,
                          savefigs: bool = False, savedir: bool = None
                          ):
        """Makes distribution comparison plots between two groups for list of variables `var_list`.
        if no group labels, uses unique values of `groupvar` (default main grouping variable) to label plot.
        --Currently not flexible for more than 2 groups, dataframe should be filtered down to two unique groups, via `filters`
        or beforehand, before running this.

        :param data: dataframe to use, defaults to scaled dataset
        :type data: pandas dataFrame, optional
        :param label: label for plot if saving figure, defaults to None
        :type label: str, optional
        :param var_list: list of variables from data to compare
        :type var_list: list
        :param groupvar: grouping variable for comparison, defaults to main grouping var
        :type groupvar: str, optional
        :param filters: dictionary to filter data before comparisons, defaults to {}
        :type filters: dict, optional
        :param groupvar_labels: labels for plot legend, defaults to unique values of grouping var
        :type groupvar_labels: list, optional
        :param rangex: x-axis range, defaults to [-2.5, 2.5]
        :type rangex: list, optional
        :param max_y: y-axis max, defaults to 50
        :type max_y: int, optional
        :param binw: bin width of density calculation, defaults to 1
        :type binw: int, optional
        :param set_size_params: (width, height) of plotted axes, in inches
        :type set_size_params: tuple of floats or ints, optional
        :param hide_text: hide all text on plot, defaults to False
        :type hide_text: bool, optional
        :param legend: show legend on plot, defaults to False
        :type legend: bool, optional
        :param user_colors: set of 2 colors for the two groups plotted, defaults to cmapset of data
        :type user_colors: list[str, str]st, optional
        :param savefigs: save figures?, defaults to False
        :type savefigs: bool, optional
        :param savedir: folder in which to save figure, defaults to None
        :type savedir: bool, optional
        """

        assert var_list is not None, '--list of variables to compare provided is empty'
        if not groupvar:    # if group parameter missing, use main one
            groupvar = self.group_var
        if data is None:  # if data param missing, use scaled dataset
            data = self.df_scaled
        label = f'{label}_' if label else ''    # add underscore to label

        for var in [col for col in data.columns if col in var_list]:
            # if type(groupvar_labels[0]) is str:
            #     x1 = np.array([val for grp, val in zip(data[groupvar], data[var]) if groupvar_labels[0] in grp])[:, np.newaxis]
            #     x2 = np.array([val for grp, val in zip(data[groupvar], data[var]) if groupvar_labels[1] in grp])[:, np.newaxis]
            # else:
            #     x1 = np.array([val for grp, val in zip(data[groupvar], data[var]) if grp == groupvar_labels[0]])[:, np.newaxis]
            #     x2 = np.array([val for grp, val in zip(data[groupvar], data[var]) if grp == groupvar_labels[1]])[:, np.newaxis]

            # dist_percent_singlevar(var, mXd, 'males')
            self.compare_dists(compvar=var, groupvar=groupvar, filters=filters, max_y=max_y, rangex=rangex,
                               binw=binw, user_colors=user_colors, user_labels=groupvar_labels, hide_text=hide_text,
                               set_size_params=set_size_params, legend=legend)
            if savefigs and not savedir:
                savedir = '.'
            save_fig(savedir, f'{label}{var}_zscoredist_F4',
                     plot_fmt='png', dpi=600, save=savefigs)

    def gmm_cluster_plot(self, grid_search, n_colors, metrics=None, hide_text=False,
                         **scatter_kwargs):

        X = self.df_scaled.loc[:, self.metric_vars].values
        if not metrics:
            metric1, metric2 = 0, 1
        else:
            metric1, metric2 = [*map(self.metric_vars.index, metrics)]

        # groupvals = scaled_df.group_vals
        groupcat = self.group_categories
        x_index = groupcat['codes']
        stylemap = {0: 'v', 1: 'o'}  # marker style for groups
        sizemap = {0: 70, 1: 60}  # marker size for groups
        grp_stylemap = [*map(stylemap.get, x_index)]
        grp_sizemap = [*map(sizemap.get, x_index)]

        color_iter = sns.color_palette("tab10", n_colors)[::-1]
        Y_ = grid_search.predict(X)

        fig, ax = plt.subplots()

        # handles for legend
        # leg_group1 = []
        # leg_group2 = []
        legend_handles = []

        for i, (mean, cov, color) in enumerate(
            zip(
                grid_search.best_estimator_.means_,
                grid_search.best_estimator_.covariances_,
                color_iter,
            )
        ):
            # v, w = linalg.eigh(cov)
            if not np.any(Y_ == i):
                continue
            x = X[Y_ == i, metric1]
            y = X[Y_ == i, metric2]

            for x_, y_, st_, si_ in zip(x, y, grp_stylemap, grp_sizemap):
                fs = plt.scatter(x_, y_, color=color,  # 'xkcd:grey',  # color,
                                 s=si_, marker=st_, label=f'Cluster {i}', **scatter_kwargs)
            legend_handles.append(fs)

            # angle = np.arctan2(w[0][1], w[0][0])
            # angle = 180.0 * angle / np.pi  # convert to degrees
            # v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            # ellipse = Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
            # ellipse.set_clip_box(fig.bbox)
            # ellipse.set_alpha(0.5)
            # ax.add_artist(ellipse)
        ax.set_xlim(-2.5, 4)
        ax.set_ylim(-2.5, 4)
        ax.set_xticks(xtx := np.linspace(-2, 4, 4))
        ax.set_xticklabels(xtx_lab := [str(tickx) for tickx in xtx])
        ax.set_yticks(xtx)
        ax.set_yticklabels(xtx_lab)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        if hide_text:
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            plt.xlabel(None)
            plt.ylabel(None)
            plt.title(None)
        else:
            plt.title(
                f"Unconstrained GMM clustering\nBest model: \n{grid_search.best_params_['covariance_type']} model, "
                f"{grid_search.best_params_['n_components']} components",
                pad=20
            )
            metric_1, metric_2 = metrics[0], metrics[1]
            plt.xlabel(f'{metric_1}')
            plt.ylabel(f'{metric_2}')
            handles = legend_handles
            labels = set(ax.get_legend_handles_labels()[1])
            ax.legend(
                handles, labels,
                loc='upper right', fontsize=8,
                bbox_to_anchor=(1.5, 1), frameon=False,
                markerscale=1,
            )
        # plt.axis("equal")
        set_size(2, 2)


class CmapSet:

    def __init__(self, color_ch, data_array, cmap_range=(0.4, 1), divs=151, donorm=True, debug=False):
        self.orig_data = data_array
        self.N = len(data_array)
        self.color_choice = color_ch
        self.divs = divs
        self.debug = debug
        self.c_debug(f'colorch: {self.color_choice}')
        retrieved_cmap = plt.cm.get_cmap(self.color_choice, divs)
        self.new_cmap = ListedColormap(
            # male color gradient
            retrieved_cmap(np.linspace(cmap_range[0], cmap_range[1], self.N)))
        if donorm:
            norm = colors.Normalize(vmin=min(data_array), vmax=max(data_array))
            self.sm = plt.cm.ScalarMappable(cmap=self.new_cmap, norm=norm)
        else:
            self.sm = plt.cm.ScalarMappable(cmap=self.new_cmap)
        try:
            self.face_colors = self.sm.to_rgba(data_array)
        except ValueError as err:
            print('could not create face_colors from data given:\n', err)

    def c_debug(self, msg):
        """
        Class implementation of check debug for printing
        :param msg:
        :param debug:
        :return:
        """
        chk_debug(msg, self.debug)

    def blend_cmap(self, name, split=True, cmap_rng: tuple[float, float] = None, col_choice=None):
        # for corr heatmaps (pca), makes it one directional - might change this
        if col_choice is None:
            col_choice = self.color_choice
        match col_choice:
            case 'viridis_r':
                tophalf, bothalf = 'BrBG_r', 'BrBG'
            case 'plasma_r':
                tophalf, bothalf = 'RdPu_r', 'RdPu'
            case _:
                if '_r' in col_choice:
                    tophalf, bothalf = f'{col_choice[:-2]}', f'{col_choice}'
                else:
                    tophalf, bothalf = f'{col_choice}_r', f'{col_choice}'

        top = cm.get_cmap(tophalf, 1024)
        bottom = cm.get_cmap(bothalf, 1024)

        if not cmap_rng:
            cmap_rng = (0, 1)
        if split:
            newcolors = np.vstack((top(np.linspace(0, 0.5, 1024)),
                                   bottom(np.linspace(0.5, 1, 1024))))
        else:
            newcolors = np.vstack((top(np.linspace(1-cmap_rng[1], 1-cmap_rng[0], 1024)),
                                   bottom(np.linspace(cmap_rng[0], cmap_rng[1], 1024))))
        if not name:
            name = f'blended_{self.color_choice}'
        blended_cmap = ListedColormap(newcolors, name=name)
        self.blended_cmap = blended_cmap


class CmapBar(CmapSet):

    def __init__(self, color_ch, cbar_array, cmap_range=(0.4, 1), divs=151, donorm=True, debug=False,
                 cvar='SI_scores', location='bottom', orientation='horizontal'):
        super().__init__(color_ch, cbar_array, cmap_range, divs, donorm, debug)
        self.c_debug(f'colorch for bar: {color_ch}')
        self.cvar = cvar
        self.c_debug(f'making colorbar from variable {cvar}')
        self.location = location
        self.orientation = orientation
        # pdb.set_trace()
        self.cbar_narray = np.linspace(min(cbar_array), max(cbar_array), divs)
        self.si_cbar = self.sm.set_array(self.cbar_narray)

# %% Running


def save_fig(path='./', label='', plot_fmt='png', dpi=400, save=False):
    if save:
        check_path(path, create_fold=True)
        plt.savefig(f'{path}/{label}.{plot_fmt}',
                    format=plot_fmt, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
        plt.clf()
