import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import os
import glob

from collections import defaultdict
from gedi.generation.hpo import get_tasks
from gedi.utils.io_helpers import get_keys_abbreviation
from gedi.utils.io_helpers import read_csvs, select_instance
from gedi.utils.param_keys import PLOT_TYPE, PROJECTION, EXPLAINED_VAR, PLOT_3D_MAP
from gedi.utils.param_keys import OUTPUT_PATH, PIPELINE_STEP
from gedi.utils.param_keys.generator import GENERATOR_PARAMS, TARGETS, PLOT_REFERENCE_FEATURE
from gedi.utils.param_keys.plotter import REAL_EVENTLOG_PATH, FONT_SIZE, BOXPLOT_WIDTH
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA


def insert_newlines(string, every=140):
    return '\n'.join(string[i:i+every] for i in range(0, len(string), every))

class MyPlotter:
    def __init__(self, interactive: bool = True, title_prefix: str = '', for_paper: bool = False):
        self.fig: Figure = Figure()
        self.axes: Axes = Axes(self.fig, [0, 0, 0, 0])
        self.interactive: bool = interactive
        self.title_prefix: str = title_prefix
        self.colors: dict = mcolors.TABLEAU_COLORS
        self.for_paper: bool = for_paper

        if self.interactive:
            mpl.use('TkAgg')

        if self.for_paper:
            self.fontsize = 18
        else:
            self.fontsize = 10

    def _set_figure_title(self):
        self.fig.suptitle(self.title_prefix)

    def _post_processing(self):
        if not self.for_paper:
            self._set_figure_title()
        plt.show()

class ModelResultPlotter(MyPlotter):
    def plot_models(self, model_results, plot_type='', plot_tics=False, components=None):
        """
        Plots the model results in 2d-coordinate system next to each other.
        Alternatively with tics of the components can be plotted under the figures when `plot_tics` is True
        :param model_results: list of dictionary
            dict should contain the keys: 'model', 'projection', 'title_prefix' (optional)
        :param plot_type: param_key.plot_type
        :param plot_tics: bool (default: False)
            Plots the component tics under the base figures if True
        :param components: int
            Number of components used for the reduced
        """
        if plot_tics:
            self.fig, self.axes = plt.subplots(components + 1, len(model_results),
                                               constrained_layout=True, figsize=(10,8))  # subplots(rows, columns)
            main_axes = self.axes[0]  # axes[row][column]
            if len(model_results) == 1:
                for component_nr in range(components + 1)[1:]:
                    self._plot_time_tics(self.axes[component_nr], model_results[DUMMY_ZERO][PROJECTION],
                                         component=component_nr)
            else:
                for i, result in enumerate(model_results):
                    df_pca = pd.DataFrame(result[PROJECTION], columns=["PC1", "PC2"])
                    sns.scatterplot(ax=self.axes[0][i], data=df_pca, x="PC1", y="PC2", palette="bright", hue=['']*len(df_pca), alpha=0.9, s=100)
                    try:
                        self.axes[0][i].set_xlabel(f"PC1 ({np.round(result[EXPLAINED_VAR][0]*100, 2)}% explained variance)")
                        self.axes[0][i].set_ylabel(f"PC2 ({np.round(result[EXPLAINED_VAR][1]*100, 2)}% explained variance)")
                    except TypeError:
                        self.axes[0][i].set_xlabel(f"TSNE_1")
                        self.axes[0][i].set_ylabel(f"TSNE_2")
                    for component_nr in range(components + 1)[1:]:
                        self._plot_time_tics(self.axes[component_nr][i], result[PROJECTION], component=component_nr)
        else:
            self.fig, self.axes = plt.subplots(1, len(model_results), constrained_layout=True)
            main_axes = self.axes

        plt.show()

    @staticmethod
    def _plot_time_tics(ax, projection, component):
        """
        Plot the time tics on a specific axis
        :param ax: axis
        :param projection:
        :param component:
        :return:
        """
        ax.cla()

        ax.set_xlabel('Time step')
        ax.set_ylabel('Component {}'.format(component))
        ax.label_outer()

        ax.plot(projection[:, component - 1])

class ArrayPlotter(MyPlotter):
    def __init__(self, interactive=False, title_prefix='', x_label='', y_label='', bottom_text=None, y_range=None,
                 show_grid=False, xtick_start=0, for_paper=False):
        super().__init__(interactive, title_prefix, for_paper)
        self.x_label = x_label
        self.y_label = y_label
        self.bottom_text = bottom_text
        self.range_tuple = y_range
        self._activate_legend = False
        self.show_grid = show_grid
        self.xtick_start = xtick_start

    def _post_processing(self, legend_outside=False):
        # self.axes.set_title(self.title_prefix)
        self.axes.set_xlabel(self.x_label, fontsize=self.fontsize)
        self.axes.set_ylabel(self.y_label, fontsize=self.fontsize)
        # plt.xticks(fontsize=self.fontsize)
        # plt.yticks(fontsize=self.fontsize)

        if self.bottom_text is not None:
            self.fig.text(0.01, 0.01, self.bottom_text, fontsize=self.fontsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(bottom=(self.bottom_text.count('\n') + 1) * 0.1)
        else:
            self.fig.tight_layout()

        if legend_outside:
            self.axes.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', fontsize=8)
            plt.subplots_adjust(bottom=0.25)
        elif self._activate_legend:
            self.axes.legend(fontsize=self.fontsize)

        if self.range_tuple is not None:
            self.axes.set_ylim(self.range_tuple)

        if self.show_grid:
            plt.grid(True, which='both')
            plt.minorticks_on()
        super()._post_processing()

    def matrix_plot(self, matrix, as_surface='2d', show_values=False):
        """
        Plots the values of a matrix on a 2d or a 3d axes
        :param matrix: ndarray (2-ndim)
            matrix, which should be plotted
        :param as_surface: str
            Plot as a 3d-surface if value PLOT_3D_MAP else 2d-axes
        :param show_values: If true, then show the values in the matrix
        """
        c_map = plt.cm.viridis
        # c_map = plt.cm.seismic
        if as_surface == PLOT_3D_MAP:
            x_coordinates = np.arange(matrix.shape[0])
            y_coordinates = np.arange(matrix.shape[1])
            x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
            self.fig = plt.figure()
            self.axes = self.fig.gca(projection='3d')
            self.axes.set_zlabel('Covariance Values', fontsize=self.fontsize)
            im = self.axes.plot_surface(x_coordinates, y_coordinates, matrix, cmap=c_map)
        else:
            self.fig, self.axes = plt.subplots(1, 1, dpi=80)
            im = self.axes.matshow(matrix, cmap=c_map)
            if show_values:
                for (i, j), value in np.ndenumerate(matrix):
                    self.axes.text(j, i, '{:0.2f}'.format(value), ha='center', va='center', fontsize=8)
        if not self.for_paper:
            self.fig.colorbar(im, ax=self.axes)
            plt.xticks(np.arange(matrix.shape[1]), np.arange(self.xtick_start, matrix.shape[1] + self.xtick_start))
            # plt.xticks(np.arange(matrix.shape[1], step=5),
            #            np.arange(self.xtick_start, matrix.shape[1] + self.xtick_start, step=5))
        self._post_processing()

    def plot_gauss2d(self,
                     x_index: np.ndarray,
                     ydata: np.ndarray,
                     new_ydata: np.ndarray,
                     gauss_fitted: np.ndarray,
                     fit_method: str,
                     statistical_function: callable = np.median):
        """
        Plot the original data (ydata), the new data (new_ydata) where the x-axis-indices is given by (x_index),
        the (fitted) gauss curve and a line (mean, median)
        :param x_index: ndarray (1-ndim)
            range of plotting
        :param ydata: ndarray (1-ndim)
            original data
        :param new_ydata: ndarray (1-ndim)
            the changed new data
        :param gauss_fitted: ndarray (1-ndim)
            the fitted curve on the new data
        :param fit_method: str
            the name of the fitting method
        :param statistical_function: callable
            Some statistical numpy function
        :return:
        """
        self.fig, self.axes = plt.subplots(1, 1, dpi=80)
        self.axes.plot(x_index, gauss_fitted, '-', label=f'fit {fit_method}')
        # self.axes.plot(x_index, gauss_fitted, ' ')
        self.axes.plot(x_index, ydata, '.', label='original data')
        # self.axes.plot(x_index, ydata, ' ')
        statistical_value = np.full(x_index.shape, statistical_function(ydata))
        if self.for_paper:
            function_label = 'threshold'
        else:
            function_label = function_name(statistical_function)
            self._activate_legend = True
        self.axes.plot(x_index, statistical_value, '-', label=function_label)
        # self.axes.plot(x_index, statistical_value, ' ')
        # self.axes.plot(x_index, new_ydata, '.', label='re-scaled data')
        self.axes.plot(x_index, new_ydata, ' ')
        self._post_processing()

    def plot_2d(self, ndarray_data, statistical_func=None):
        self.fig, self.axes = plt.subplots(1, 1)
        self.axes.plot(ndarray_data, '-')
        if statistical_func is not None:
            statistical_value = statistical_func(ndarray_data)
            statistical_value_line = np.full(ndarray_data.shape, statistical_value)
            self.axes.plot(statistical_value_line, '-',
                           label=f'{function_name(statistical_func)}: {statistical_value:.4f}')
        self._activate_legend = False
        self._post_processing()

    def plot_merged_2ds(self, ndarray_dict: dict, statistical_func=None):
        self.fig, self.axes = plt.subplots(1, 1, dpi=80)
        self.title_prefix += f'with {function_name(statistical_func)}' if statistical_func is not None else ''
        for key, ndarray_data in ndarray_dict.items():
            # noinspection PyProtectedMember
            color = next(self.axes._get_lines.prop_cycler)['color']
            if statistical_func is not None:
                if isinstance(ndarray_data, list):
                    ndarray_data = np.asarray(ndarray_data)
                self.axes.plot(ndarray_data, '-', color=color)
                statistical_value = statistical_func(ndarray_data)
                statistical_value_line = np.full(ndarray_data.shape, statistical_value)
                self.axes.plot(statistical_value_line, '--',
                               label=f'{key.strip()}: {statistical_value:.4f}', color=color)
            else:
                self.axes.plot(ndarray_data, '-', color=color, label=f'{key.strip()[:35]}')

        self._activate_legend = True
        self._post_processing()

class BenchmarkPlotter:
    def __init__(self, benchmark_results, output_path = None):
        self.plot_miners_correlation(benchmark_results, output_path=output_path)
        self.plot_miner_feat_correlation(benchmark_results, output_path=output_path)
        self.plot_miner_feat_correlation(benchmark_results, mean='methods', output_path=output_path)

    def plot_miner_feat_correlation(self, benchmark, mean='metrics', output_path=None):
        df = benchmark.loc[:, benchmark.columns!='log']
        corr = df.corr()

        if mean == 'methods':
            for method in ['inductive', 'heu', 'ilp']:
                method_cols = [col for col in corr.columns if col.startswith(method)]
                corr[method+'_avg'] = corr.loc[:, corr.columns.isin(method_cols)].mean(axis=1)
        elif mean == 'metrics':
            for metric in ['fitness', 'precision', 'generalization', 'simplicity']:
                metric_cols = [col for col in corr.columns if col.endswith(metric)]
                corr[metric+'_avg'] = corr.loc[:, corr.columns.isin(metric_cols)].mean(axis=1)

        avg_cols = [col for col in corr.columns if col.endswith('_avg')]

        benchmark_result_cols = [col for col in corr.columns if col.startswith('inductive')
                                or col.startswith('heu') or col.startswith('ilp')]

        corr = corr[:][~corr.index.isin(benchmark_result_cols)]

        fig, axes = plt.subplots( 1, len(avg_cols), figsize=(15,10))

        for i, ax in enumerate(axes):
            cbar = True if i==3 else False
            corr = corr.sort_values(avg_cols[i], axis=0, ascending=False)
            b= sns.heatmap(corr[[avg_cols[i]]][:],
                        ax=ax,
                        xticklabels=[avg_cols[i]],
                        yticklabels=corr.index,
                        cbar=cbar)
        plt.subplots_adjust(wspace = 1, top=0.9, left=0.15)
        fig.suptitle(f"Feature and performance correlation per {mean.split('s')[0]} for {len(benchmark)} event-logs")
        if output_path != None:
            output_path = output_path+f"/minperf_corr_{mean.split('s')[0]}_el{len(benchmark)}.jpg"
            fig.savefig(output_path)
            print(f"SUCCESS: Saved correlation plot at {output_path}")
        #plt.show()

    def plot_miners_correlation(self, benchmark, output_path=None):
        benchmark_result_cols = [col for col in benchmark.columns if col.startswith('inductive')
                                or col.startswith('heu') or col.startswith('ilp')]
        df = benchmark.loc[:, benchmark.columns!='log']
        df = df.loc[:, df.columns.isin(benchmark_result_cols)]

        corr = df.corr()
        fig, ax = plt.subplots(figsize=(15,10))
        b= sns.heatmap(corr,
                    ax=ax,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values)
        plt.title(f"Miners and performance correlation for {len(benchmark)} event-logs", loc='center')
        if output_path != None:
            output_path = output_path+f"/minperf_corr_el{len(benchmark)}.jpg"
            fig.savefig(output_path)
            print(f"SUCCESS: Saved correlation plot at {output_path}")
        #plt.show()

class FeaturesPlotter:
    def __init__(self, features, params=None):
        output_path = params[OUTPUT_PATH] if OUTPUT_PATH in params else None
        plot_type = f", plot_type='{params[PLOT_TYPE]}'" if params.get(PLOT_TYPE) else ""
        font_size = f", font_size='{params[FONT_SIZE]}'" if params.get(FONT_SIZE) else ""
        boxplot_w = f", boxplot_w='{params[BOXPLOT_WIDTH]}'" if params.get(BOXPLOT_WIDTH) else ""
        LEGEND = ", legend=True" if params.get(PIPELINE_STEP) else ""

        source_name = os.path.split(params['input_path'])[-1].replace(".csv", "")+"_"
        #output_path = os.path.join(output_path, source_name)
        if REAL_EVENTLOG_PATH in params:
            real_eventlogs_path=params[REAL_EVENTLOG_PATH]
            real_eventlogs = pd.read_csv(real_eventlogs_path)
            fig, output_path = eval(f"self.plot_violinplot_multi(features, output_path, real_eventlogs, source='{source_name}' {plot_type}{font_size}{boxplot_w}{LEGEND})")
        else:
            fig, output_path = eval(f"self.plot_violinplot_single(features, output_path, source='{source_name}' {plot_type}{font_size}{boxplot_w})")

        if output_path != None:
            os.makedirs(os.path.split(output_path)[0], exist_ok=True)
            fig.savefig(output_path)
            print(f"SUCCESS: Saved {plot_type} plot in {output_path}")


    def plot_violinplot_single(self, features, output_path=None, source="_",  plot_type="violinplot", font_size=16, boxplot_w=16):
        columns = features.columns[1:]
        df1=features.select_dtypes(exclude=['object'])

        fig, axes = plt.subplots(len(df1.columns),1, figsize=(int(boxplot_w),len(df1.columns)))
        for i, ax in enumerate(axes):
                eval(f"sns.{plot_type}(data=df1, x=df1[df1.columns[i]], ax=ax)")
        fig.suptitle(f"{len(columns)} features distribution for {len(features)} generated event-logs", fontsize=font_size, y=1)
        fig.tight_layout()


        output_path=output_path+f"/{plot_type}s_{source}{len(columns)}fts_{len(df1)}gEL.jpg"

        return fig, output_path

    def plot_violinplot_multi(self, features, output_path, real_eventlogs, source="_", plot_type="violinplot",
                              font_size=24, legend=False, boxplot_w=16):
        LOG_NATURE = "Log Nature"
        GENERATED = "Generated"
        REAL = "Real"
        FONT_SIZE=font_size
        alpha = 0.7
        color = sns.color_palette("bright")
        markers = ['o','X']
        inner_param = ''

        features[LOG_NATURE] = GENERATED
        real_eventlogs[LOG_NATURE] = REAL

        bdf = pd.concat([features, real_eventlogs])
        bdf = bdf[features.columns]
        bdf = bdf.dropna(axis='rows')

        columns = bdf.columns[3:]
        dmf1=bdf.select_dtypes(exclude=['object'])

        if plot_type == 'violinplot':
            inner_param = 'inner = None,'

        fig, axes = plt.subplots(len(dmf1.columns),1, figsize=(int(boxplot_w),len(dmf1.columns)*1.25), dpi=300)
        if isinstance(axes, Axes): # not isinstance(axes, list):
            axes = [axes]
        #nature_types = set(['Generated', 'Real'])#set(bdf['Log Nature'].unique())
        nature_types = list(reversed(bdf['Log Nature'].unique()[:2]))
        for i, ax in enumerate(axes):
            for j, nature in enumerate(nature_types):
                eval(f"sns.{plot_type}(data=bdf[bdf['Log Nature']==nature], x=dmf1.columns[i], palette=[color[j]], {inner_param} ax=ax)")
                eval(f"sns.stripplot(data=bdf[bdf['Log Nature']==nature], x=dmf1.columns[i], palette=[color[j]], marker=markers[j], {inner_param} ax=ax)")
            for collection in ax.collections:
                collection.set_alpha(alpha)

            for patch in ax.patches:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, alpha))

            custom_lines = [
                Line2D([0], [0], color=color[nature], lw=4, alpha=alpha)
                for nature in [0,1,2]
            ]
            #ax.legend(custom_lines, bdf['Log Nature'].unique(), title= "Log Nature")
            #sns.set_context("paper", font_scale=1.5)
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
            ax.tick_params(axis='both', which='minor', labelsize=FONT_SIZE)
            ax.set_xlabel(dmf1.columns[i], fontsize=FONT_SIZE)


        if legend:
            fig.legend(custom_lines, nature_types, loc='upper right', ncol=len(nature_types), prop={'size': FONT_SIZE})
            plt.legend(fontsize=FONT_SIZE)
        #fig.suptitle(f"{len(features.columns)-2} features distribution for {len(real_eventlogs[real_eventlogs['Log Nature'].isin(nature_types)])} real and {len(features)} generated event-logs", fontsize=16, y=1)
        plt.yticks(fontsize=FONT_SIZE)
        plt.xticks(fontsize=FONT_SIZE)

        fig.tight_layout()

        output_path = output_path+f"/{plot_type}s_{source}{len(columns)}fts_{len(features)}gEL_of{len(bdf[bdf['Log Nature'].isin(nature_types)])}.jpg"
        return fig, output_path

class AugmentationPlotter(object):
    """Plotter for the augmented features.
    If just 2 features are examined, the plotter outputs a scatterplot with the two features defining
    the dimensions.
    IF more than 2 features are examined, a PCA is performed first before the first two principal
    components are plotted.

    Parameters
    ----------
    features : pd.DataFrame
        dataFrame containing the information of the real and synthesized datasets.
    """

    def __init__(self, features, params=None) -> None:
        output_path = params[OUTPUT_PATH] if OUTPUT_PATH in params else None
        self.sampler = params['augmentation_params']['method']
        eval(f"self.plot_augmented_features(features, output_path)")


    def plot_augmented_features(self, features, output_path=None) -> None:
        """Plotting for augmented features. When more than 2 features are selected, the
        plot will show the result after applying a PCA; otherwise the 2 features are
        plotted according to the values.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame containing the augmented features
        output_path : str, optional
            Path to the output file, by default None
        """
        if len(features.all.columns) < 2:
            raise AssertionError ("AugmentationPlotter - More than 2 (augmented) features are expected for plotting.")

        if len(features.all.columns) > 2:
            self._plot_pca(features, output_path)
        else:
            self._plot_2d(features, output_path)


    def _plot_2d(self, features, output_path=None) -> None:
        """Fnc for plotting 2D features without any dimension reduction technique being applied.

        Parameters
        ----------
        features : pd.DataFrame
            Dataframe containing the augmented features
        output_path : str, optional
            Path to the output file, by default None
        """
        col1_name, col2_name = features.all.columns

        # INIT - settings
        X = features.all.iloc[:-features.new_samples.shape[0]]
        X = X.to_numpy()
        X_aug = features.all.to_numpy()
        sns.set_theme()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f'Log Descriptors - real: {X.shape[0]}, synth.: {X_aug.shape[0]-X.shape[0]}', fontsize=16)

        # Normalizer: applied to each observation -> row values have unit norm
        normalizer = Normalizer(norm="l2").fit(X)
        normed_data = normalizer.transform(X_aug)

        # StandardScaler: applied to features -> col values have unit norm
        scaler = StandardScaler().fit(X)
        scaled_data = scaler.transform(X_aug)

        # PLOT - raw 2d data
        X_aug = self._add_real_synth_encoding(X_aug, X, X_aug)
        df_raw = self._convert_to_df(X_aug, [col1_name, col2_name, 'type'])
        sns.scatterplot(ax=ax1, data=df_raw, x=col1_name, y=col2_name, palette="bright", 
                        hue = "type", alpha=0.5, s=100).set_title("Raw data") 
        ax1.get_legend().set_title("")

        # PLOT - normed 2d data
        normed_data = self._add_real_synth_encoding(normed_data, X, X_aug)
        df_normed = self._convert_to_df(normed_data, [col1_name, col2_name, 'type'])
        sns.scatterplot(ax=ax2, data=df_normed, x=col1_name, y=col2_name, palette="bright", 
                        hue = 'type', alpha=0.5, s=100).set_title("Normalized data")
        ax2.get_legend().set_title("")

        # PLOT - scaled 2d data
        scaled_data = self._add_real_synth_encoding(scaled_data, X, X_aug)
        df_scaled = self._convert_to_df(scaled_data, [col1_name, col2_name, 'type'])
        sns.scatterplot(ax=ax3, data=df_scaled, x=col1_name, y=col2_name, palette="bright", 
                        hue = 'type', alpha=0.5, s=100).set_title("Scaled data")
        ax3.get_legend().set_title("")

        plt.tight_layout()

        # OUTPUT
        if output_path != None:
            output_path += f"/augmentation_2d_plot_{col1_name}-{col2_name}_{self.sampler}.jpg"
            fig.savefig(output_path)
            print(f"SUCCESS: Saved augmentation pca plot at {output_path}")

    def _add_real_synth_encoding(self, arr, X, X_aug) -> np.array:
        """Helper function for adding one additional column to the array in the last column. 
        The last column indicates whether it is a real data (=0) or synthesized (=1).

        Parameters
        ----------
        arr : np.array
            data array
        X : np.array
            data of real datasets
        X_aug : np.array
            data of real datasets and synthesized datasets

        Returns
        -------
        np.array
            array containing the data with an additional last column indicating whether the
            data comes from a real dataset or synthesized one
        """
        real_synth_enc = np.array([0]*X.shape[0] + [1]*(X_aug.shape[0]-X.shape[0])).reshape(-1, 1)
        return np.hstack ([arr, real_synth_enc])

    def _convert_to_df(self, arr, colnames, enc=['real', 'synth']) -> pd.DataFrame:
        """Converts the attached array to a dataframe. The column names are
        defined by the respective parameters, where the last column is encoded
        by the string array of the enc parameter.

        Parameters
        ----------
        arr : np.array
            _description_
        colnames : list
            column names of returned dataframe
        enc : list, optional
            labels for real vs. generated data, by default ['real', 'synth']

        Returns
        -------
        pd.DataFrame
            dataframe containing the attached data array with encoded values in the last column 
        """
        df = pd.DataFrame(arr, columns=colnames)
        df.loc[df.iloc[:, -1] == 0, colnames[-1]] = enc[0]
        df.loc[df.iloc[:, -1] == 1, colnames[-1]] = enc[1]
        return df

    def _plot_pca(self, features, output_path=None) -> None:
        """Fnc for plotting features with PCA as dimension reduction technique being applied.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame containing the augmented features
        output_path : str, optional
            path to output file, by default None
        """
        # INIT - settings
        n_features = features.all.shape[1]
        X = features.all.iloc[:-features.new_samples.shape[0]]
        X = X.to_numpy()
        X_aug = features.all.to_numpy()
        sns.set_theme()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f'Log Descriptors - real: {X.shape[0]}, synth.: {X_aug.shape[0]-X.shape[0]}', fontsize=16)

        pca_components = 2
        pca = PCA(n_components=pca_components)

        # Normalizer: applied to each observation -> row values have unit norm
        normalizer = Normalizer(norm="l2").fit(X)
        normed_data_real = normalizer.transform(X)
        normed_data_aug = normalizer.transform(X_aug)

        # StandardScaler: applied to features -> col values have unit norm
        scaler = StandardScaler().fit(X)
        scaled_data_real = scaler.transform(X)
        scaled_data_aug = scaler.transform(X_aug)

        # PLOT - PCA on raw input
        fit_pca = pca.fit(X)
        X_new = fit_pca.transform(X_aug)
        X_new = self._add_real_synth_encoding(X_new[:, :pca_components], X, X_aug)
        df_pca = self._convert_to_df(X_new, ['PC_1', 'PC_2', 'type'])
        sns.scatterplot(ax=ax1, data=df_pca, x="PC_1", y="PC_2", palette="bright", hue = 'type', alpha=0.5, s=100)
        ax1.set_xlabel(f"PC1 ({np.round(pca.explained_variance_ratio_[0]*100, 2)}% explained variance)")
        ax1.set_ylabel(f"PC2 ({np.round(pca.explained_variance_ratio_[1]*100, 2)}% explained variance)")
        ax1.get_legend().set_title("")

        # PLOT - PCA on normed data
        fit_norm_pca = pca.fit(normed_data_real)
        X_new_normed = fit_norm_pca.transform(normed_data_aug)
        X_new_normed = self._add_real_synth_encoding(X_new_normed[:, :pca_components], X, X_aug)
        df_pca_normed = self._convert_to_df(X_new_normed, ['PC_1', 'PC_2', 'type'])
        sns.scatterplot(ax=ax2, data=df_pca_normed, x="PC_1", y="PC_2", palette="bright", hue = 'type', alpha=0.5, s=100)
        ax2.set_xlabel(f"PC1 ({np.round(pca.explained_variance_ratio_[0]*100, 2)}% explained variance)")
        ax2.set_ylabel(f"PC2 ({np.round(pca.explained_variance_ratio_[1]*100, 2)}% explained variance)")
        ax2.get_legend().set_title("")

        # PLOT - PCA on scaled data
        fit_sca_pca = pca.fit(scaled_data_real)
        X_new_sca = fit_sca_pca.transform(scaled_data_aug)
        X_new_sca = self._add_real_synth_encoding(X_new_sca[:, :pca_components], X, X_aug)
        df_pca_scaled = self._convert_to_df(X_new_sca,  ['PC_1', 'PC_2', 'type'])
        sns.scatterplot(ax=ax3, data=df_pca_scaled, x="PC_1", y="PC_2", palette="bright", hue = 'type', alpha=0.5, s=100)
        ax3.set_xlabel(f"PC1 ({np.round(pca.explained_variance_ratio_[0]*100, 2)}% explained variance)")
        ax3.set_ylabel(f"PC2 ({np.round(pca.explained_variance_ratio_[1]*100, 2)}% explained variance)")
        ax3.get_legend().set_title("")

        plt.tight_layout()

        # OUTPUT
        if output_path != None:
            output_path += f"/augmentation_pca_{n_features}_{self.sampler}.jpg"
            fig.savefig(output_path)
            print(f"SUCCESS: Saved augmentation pca plot at {output_path}")


class GenerationPlotter(object):
    def __init__(self, gen_cfg, model_params, output_path, input_path=None):
        print(f"Running plotter for {len(gen_cfg)} genEL, params {model_params}, output path: {output_path}")
        self.output_path = output_path
        self.input_path = input_path
        self.model_params = model_params
        if gen_cfg.empty: # Deactivated for tests
            return
        if "features" in gen_cfg.columns:
            self.gen = gen_cfg.features
            self.gen=pd.concat([pd.DataFrame.from_dict(entry, orient="Index").T for entry in self.gen]).reset_index(drop=True)
        else:
            self.gen = gen_cfg.reset_index(drop=True)

        if GENERATOR_PARAMS in model_params:
            self.tasks, _ = get_tasks(model_params[GENERATOR_PARAMS][TARGETS])
            feature_list = list(self.tasks.select_dtypes(exclude=['object']).keys())
            ref_feat = None
            if PLOT_REFERENCE_FEATURE in model_params[GENERATOR_PARAMS]and model_params[GENERATOR_PARAMS][PLOT_REFERENCE_FEATURE] != "":
                ref_feat = model_params[GENERATOR_PARAMS][PLOT_REFERENCE_FEATURE]
            reference_feature_list = feature_list if ref_feat is None else [ref_feat]

        self.plot_settings()

        if input_path is not None:
            # plot single reference feature compared to values stored in .csvs
            if isinstance(input_path, str) and input_path.endswith(".csv"):
                f_d = pd.read_csv(input_path)
                f_d = {model_params['reference_feature']: f_d}
            elif isinstance(input_path, list):
                self.plot_dist_mx(model_params)
            else:
                f_d = read_csvs(input_path, model_params['reference_feature'])
                tasks, _ = get_tasks(model_params['targets'], reference_feature=model_params['reference_feature'])
                self.plot_reference_feature_plot(tasks, f_d, model_params['reference_feature'])
        else:
            # start all plotting procedures at once
            self.plot_feat_comparison(feature_list, reference_feature_list)


    def plot_reference_feature_plot(self, orig_targets, f_dict, reference_feature, resolution=10):
        fig1, axes = plt.subplots(1, len(f_dict), figsize=(20, 4))
        if isinstance(axes,Axes):
            axes = [axes]
        fig2, axes_mesh = plt.subplots(1, len(f_dict), figsize=(20, 4), layout='compressed')
        if isinstance(axes_mesh, Axes):
            axes_mesh = [axes_mesh]

        for idx_ax, (k, v) in enumerate(f_dict.items()):
            if isinstance(orig_targets, pd.DataFrame):
                targets = orig_targets.copy()
            elif isinstance(orig_targets, defaultdict):
                if k not in orig_targets:
                    print(f"[WARNING] {k} not in targets. Only in generated features. Will continue with next feature to compare with")
                    continue
                targets = orig_targets[k].copy()
            else:
                print(f"[ERR] Unknown file format for targets {type(orig_targets)}. Close program (Exit Code: 0).")

            # Identify NAN values of reference feature
            target_nan_values_idx_reference = np.where(targets[reference_feature].isna())[0]
            target_nan_logs_reference = targets.loc[target_nan_values_idx_reference]['log']
            # Identify NAN values of competitor feature
            target_nan_values_idx_competitor = np.where(targets[k].isna())[0]
            target_nan_logs_competitor = targets.loc[target_nan_values_idx_competitor]['log']
            # Collection of indices to drop
            target_nan_indices = np.unique(np.concatenate((target_nan_values_idx_competitor, target_nan_values_idx_reference)))
            # Drop NAN values in targets DataFrame
            targets.drop(axis='index', index=target_nan_indices, inplace=True)

            # Check for indices in generated DataFrame
            reference_values_idx_reference = v[v['log'].isin(list(target_nan_logs_reference))].index
            reference_values_idx_competitor = v[v['log'].isin(list(target_nan_logs_competitor))].index
            # Collection of indices to drop for reference
            reference_nan_indices = np.unique(np.concatenate((reference_values_idx_reference, reference_values_idx_competitor)))
            # Drop NAN values in generated DataFrame
            v.drop(axis='index', index=reference_nan_indices, inplace=True)

            # Plot generated DataFrame + targets DataFrame
            v.plot.scatter(x=v.columns.get_loc(reference_feature), y=v.columns.get_loc(k), ax=axes[idx_ax], c="red", alpha=0.3)
            targets.plot.scatter(x=targets.columns.get_loc(reference_feature), y=targets.columns.get_loc(k), ax=axes[idx_ax], c='blue', alpha=0.3)

            Z = np.zeros([resolution+1, resolution+1])
            cnt_Z = np.zeros([resolution+1, resolution+1])
            Z.fill(np.nan)

            min_Z_X = np.min(targets[reference_feature])
            min_Z_Y = np.min(targets[k])
            max_Z_X = np.max(targets[reference_feature])
            max_Z_Y = np.max(targets[k])
            step_Z_X = np.round((max_Z_X - min_Z_X) / float(resolution), 4)
            step_Z_Y = np.round((max_Z_Y - min_Z_Y) / float(resolution), 4)

            cum_sum=0
            for idx in v.index:
                if isinstance(v, pd.DataFrame) and 'log' in v.columns:
                    c_log = v.loc[idx, 'log']
                    if c_log in targets['log'].values:
                        gen_entry = targets[targets['log'] == c_log]
                    else:
                        print(f"INFO: no value for {c_log} in generated files.")
                        gen_entry = targets
                else:
                    gen_entry = targets

                # Plot connection line 
                axes[idx_ax].plot([v[reference_feature][idx], gen_entry[reference_feature].values[0]],
                        [v[k][idx], gen_entry[k].values[0]],
                        c="green", alpha=0.25)
                # Plot textual annotation
                axes[idx_ax].annotate(gen_entry['log'].values[0], 
                                      (gen_entry[reference_feature].values[0], gen_entry[k].values[0]), 
                                      fontsize=5)

                # Compute distance between real and generated dot
                vec1 = np.array([v[reference_feature][idx], v[k][idx]])
                vec2 = np.array([gen_entry[reference_feature].values[0], gen_entry[k].values[0]])

                Z_idx = int (np.round((gen_entry[reference_feature].values[0] - min_Z_X) / step_Z_X))
                Z_idy = int (np.round((gen_entry[k].values[0] - min_Z_Y) / step_Z_Y))
                if np.isnan(Z[Z_idx][Z_idy]):
                    Z[Z_idx][Z_idy] = 0.0
                Z[Z_idx][Z_idy] += np.linalg.norm(vec1 - vec2)
                cnt_Z[Z_idx][Z_idy] += 1

                cum_sum += np.linalg.norm(vec1 - vec2)

            print(f"INFO: Cumulated distances objectives <-> generated features for '{reference_feature}' vs. '{k}': {cum_sum:.4f}")

            X, Y = np.meshgrid(np.linspace(min_Z_X, max_Z_X, resolution+1), 
                               np.linspace(min_Z_Y, max_Z_Y, resolution+1))
            cmap = plt.colormaps['viridis_r']
            Z[np.isnan(Z)] = np.sqrt(2) 
            cnt_Z[cnt_Z==0] = 1
            Z /= cnt_Z
            colormesh = axes_mesh[idx_ax].pcolormesh(X, Y, Z.T, shading='nearest', cmap=cmap)

            axes_mesh[idx_ax].set_xlabel(reference_feature)
            axes_mesh[idx_ax].set_ylabel(k)
            if idx_ax == (len(f_dict)-1):
                cbar = fig2.colorbar(colormesh, ax=axes_mesh, orientation='vertical', pad=0.01)
                cbar.ax.set_ylabel('Feature dist. of generated EDs and objectives',fontsize=8, rotation=90, labelpad=-50)
            axes[idx_ax].set_title(f"Cumulated distances {cum_sum:.4f}")

        tasks_keys = f_dict.keys()
        tasks_keys = list(sorted(tasks_keys))
        abbreviations = get_keys_abbreviation(tasks_keys)
        ref_short_name = get_keys_abbreviation([reference_feature])

        fig1_title = f'Feature Comparison - {reference_feature}'
        fig1.suptitle(fig1_title, fontsize=6)
        fig1.tight_layout()
        distance_plot_path = os.path.join(self.output_path,
                                          f"plot_genEL{len(self.gen)}_tasks{len(tasks_keys)}_{ref_short_name}_vs_{abbreviations}.png")
        fig1.savefig(distance_plot_path)
        print(f"Saved objectives vs. genEL features plot in {distance_plot_path}")

        fig2.suptitle(f'Meshgrid Comparison - {reference_feature}', fontsize=6)
        meshgrid_plot_path = os.path.join(self.output_path,
                                  f"plot_meshgrid_genEL{len(self.gen)}_tasks{len(tasks_keys)}_{ref_short_name}_vs_{abbreviations}.png")

        fig2.savefig(meshgrid_plot_path)
        print(f"Saved meshgrid plot in {meshgrid_plot_path}")


    def plot_single_comparison(self, tasks, objective1, objective2, ax, ax_cmesh, fig2, axes_meshes, flag_plt_clbar):
        if len(tasks.select_dtypes(include=['object']).columns)==0:
            tasks['task']=[f"task_{str(x+1)}" for x in tasks.index.values.tolist()]
        id_col = tasks.select_dtypes(include=['object']).dropna(axis=1).columns[0]
        tasks.plot.scatter(x=objective1, y=objective2, ax=ax, alpha=0.3)
        self.gen.plot.scatter(x=objective1, y=objective2, c="red", ax=ax, alpha=0.3)

        Z = np.zeros([tasks[objective1].unique().size, tasks[objective2].unique().size])
        cnt_Z = np.zeros([tasks[objective1].unique().size, tasks[objective2].unique().size])
        Z.fill(np.inf)
        cum_sum = 0
        for idx in tasks.index:
            if isinstance(tasks, pd.DataFrame) and 'log' in tasks.columns:
                c_log = tasks.loc[idx, 'log']
                if c_log in self.gen['log'].values:
                    gen_entry = self.gen[self.gen['log'] == c_log]
                else:
                    print(f"INFO: no value for {c_log} in generated files.")
                    gen_entry = self.gen
            else:
                gen_entry = self.gen

            ax.plot([tasks[objective1][idx], gen_entry[objective1].values[0]],
                    [tasks[objective2][idx], gen_entry[objective2].values[0]],
                    c="green", alpha=0.25)

            ax.annotate(tasks[id_col][idx], (tasks[objective1][idx], tasks[objective2][idx]), fontsize=5)

            vec1 = np.array([tasks[objective1][idx], tasks[objective2][idx]])
            vec2 = np.array([gen_entry[objective1].values[0], gen_entry[objective2].values[0]])

            Z_idx = np.where(tasks[objective1].unique() == tasks[objective1][idx])[0][0]
            Z_idy = np.where(tasks[objective2].unique() == tasks[objective2][idx])[0][0]
            if np.isinf(Z[Z_idx][Z_idy]):
                Z[Z_idx][Z_idy] = 0.0
            Z[Z_idx][Z_idy] += np.linalg.norm(vec1 - vec2)
            cnt_Z[Z_idx][Z_idy] += 1 
            cum_sum += np.linalg.norm(vec1 - vec2)

        print(f"INFO: Cumulated distances objectives <-> generated features for '{objective1}' vs. '{objective2}':", cum_sum)
        ax.set_title(f"Cumulated distances {cum_sum:.4f}")
        X, Y = np.meshgrid(tasks[objective1].unique(), tasks[objective2].unique())
        cmap = plt.colormaps['viridis_r']

        Z[np.isinf(Z)] = np.sqrt(2)
        cnt_Z[cnt_Z==0] = 1
        Z /= cnt_Z

        colormesh = ax_cmesh.pcolormesh(X, Y, Z.T, shading='nearest', cmap=cmap) # vmin=0.0, vmax=1.0, cmap=cmap)
        ax_cmesh.set_xlabel(objective1)
        ax_cmesh.set_ylabel(objective2)
        if flag_plt_clbar:
            fig2.colorbar(colormesh, ax=axes_meshes, orientation='vertical')
        return colormesh


    def plot_settings(self):
        mpl.rc('axes', titlesize=8)  # fontsize of the axes title
        mpl.rc('axes', labelsize=8)  # fontsize of the x and y labels
        mpl.rc('font', size=8)


    def plot_feat_comparison(self, feature_list, reference_list):
        len_features = len(feature_list)
        len_ref_feats = len(reference_list)
        fig1, axes = plt.subplots(len_ref_feats, len_features)
        fig2, axes_meshes = plt.subplots(len_ref_feats, len_features, layout='compressed')

        for idx1, entry1 in enumerate(reference_list):
            for idx2, entry2 in enumerate(feature_list):
                if isinstance(axes, Axes):
                    ax = axes
                    ax_cmesh = axes_meshes
                elif len_ref_feats == 1:
                    ax = axes[idx2]
                    ax_cmesh = axes_meshes[idx2]
                else:
                    ax = axes[idx1][idx2]
                    ax_cmesh = axes_meshes[idx1][idx2]
                flag_plt_clbar = False
                if ((idx2 == (len(feature_list)-1)) & (idx1 == len(reference_list)-1)):
                    flag_plt_clbar = True
                colormesh = self.plot_single_comparison(self.tasks, entry1, entry2, ax, ax_cmesh, fig2, axes_meshes, flag_plt_clbar)

        objectives_keys = self.tasks.select_dtypes(exclude=['object']).columns
        objectives_keys = list(sorted(objectives_keys))
        abbreviations = get_keys_abbreviation(objectives_keys)

        fig1_title = f'Feature Comparison with {self.model_params[GENERATOR_PARAMS]}'
        fig1.suptitle(insert_newlines(fig1_title), fontsize=6)
        fig1.tight_layout()
        distance_plot_path = os.path.join(self.output_path,
                                          f"eval_genEL{len(self.gen)}_objectives{len(objectives_keys)}_trials{self.model_params['generator_params']['n_trials']}_{abbreviations}.png")
        os.makedirs(self.output_path, exist_ok=True)
        fig1.savefig(distance_plot_path)
        print(f"Saved objectives vs. genEL features plot in {distance_plot_path}")

        # fig2.suptitle('Meshgrid Comparison', fontsize=12)
        meshgrid_plot_path = os.path.join(self.output_path,
                                  f"meshgrid_genEL{len(self.gen)}_objectives{len(objectives_keys)}_trials{self.model_params['generator_params']['n_trials']}_{abbreviations}.png")

        fig2.savefig(meshgrid_plot_path)
        print(f"Saved meshgrid plot in {meshgrid_plot_path}")


    def plot_dist_mx (self, model_params):
        gen_dict = defaultdict(lambda: defaultdict(dict))
        targets_dict = defaultdict(lambda: defaultdict(dict))

        set_ = set()
        for in_file in self.input_path:
            for file in glob.glob(f'{in_file}*.csv'):
                read_in = pd.read_csv(file)
                feat1, feat2 = None, None
                if len(read_in.columns) == 2:
                    feat1 = read_in.columns[0]
                    feat2 = feat1
                else:
                    feat1 = read_in.columns[0]
                    feat2 = read_in.columns[1]
                read_in['fn'] = file
                gen_dict[feat1][feat2] = read_in
                set_.add(feat1)
                set_.add(feat2)
        for target_file in model_params["targets"]:
            for file in glob.glob(f'{target_file}*.csv'):
                read_in = pd.read_csv(file)
                if 'task' in read_in.columns:
                    read_in.rename(columns={"task":"log"}, inplace=True)
                feat1, feat2 = None, None
                if len(read_in.columns) == 2:
                    feat1 = read_in.columns[1]
                    feat2 = feat1
                else:
                    feat1 = read_in.columns[1]
                    feat2 = read_in.columns[2]
                read_in['fn'] = file
                targets_dict[feat1][feat2] = read_in
                set_.add(feat1)
                set_.add(feat2)

        keys = sorted(list(set_))
        result_df = pd.DataFrame(index=keys, columns=keys)

        dist_list = list()

        for gen_idx, (gen_obj1_key, gen_obj1_vals) in enumerate(gen_dict.items()):
            if gen_obj1_key not in targets_dict:
                continue

            for gen_obj1_value in gen_obj1_vals:
                if gen_obj1_value not in targets_dict[gen_obj1_key]:
                    continue

                gen_df = gen_dict[gen_obj1_key][gen_obj1_value]
                target_df = targets_dict[gen_obj1_key][gen_obj1_value]


                cnt = 0
                cum_sum = 0
                for i in gen_df.index:
                    current_log_name = gen_df.loc[i, 'log']
                    if current_log_name in target_df['log'].values:
                        target_entry = target_df[target_df['log'] == current_log_name]
                    else:
                        print (f"[INFO] no value found for {current_log_name} in target file")

                    vec1 = np.array([gen_df[gen_obj1_key][i], gen_df[gen_obj1_value][i]])
                    vec2 = np.array([target_entry[gen_obj1_key].values[0], target_entry[gen_obj1_value].values[0]])

                    cum_sum += np.linalg.norm(vec1 - vec2)
                    cnt += 1

                    THRESHOLD=0.1
                    if np.linalg.norm(vec1 - vec2) < THRESHOLD and len(gen_df.columns)>3:#3 for 1 objective
                        path_splits = gen_df.loc[i, 'fn'].split("/")
                        data_splits = path_splits[-1][:-4].split("_")
                        log_path= f'grid_2objectives_{data_splits[1]}_{data_splits[2]}/2_{data_splits[1]}_{data_splits[2]}/genEL{current_log_name}_*.xes'
                        dest, len_is = select_instance(in_file.replace("features/", ""), log_path)

                        dist_list.append(np.linalg.norm(vec1 - vec2))

                cum_sum /= cnt

                result_df.loc[gen_obj1_key, gen_obj1_value] = cum_sum
                result_df.loc[gen_obj1_value, gen_obj1_key] = cum_sum
        try:
            print(f"INFO: Instance selection saved {len_is} ED selection in {dest}")
        except UnboundLocalError as e:
            print(e)
        ratio_most_common_variant = 2.021278 / 11.0
        ratio_top_10_variants = 0.07378 / 11.0
        ratio_variants_per_number_of_traces = 0.016658 / 11.0
        result_df['ratio_most_common_variant']['ratio_most_common_variant'] = ratio_most_common_variant
        result_df['ratio_top_10_variants']['ratio_top_10_variants'] = ratio_top_10_variants
        result_df['ratio_variants_per_number_of_traces']['ratio_variants_per_number_of_traces'] = ratio_variants_per_number_of_traces

        abbrvs_key = get_keys_abbreviation(keys)
        result_df.columns = abbrvs_key.split("_")
        result_df.index = abbrvs_key.split("_")
        # result__mx = result_df.values.astype(np.float16)
        # result__mx[np.isnan(result__mx)] = 0
        img = sns.heatmap(result_df.astype(np.float16),annot=True, cmap="viridis_r", vmin=0.0, vmax=1.0)
        # plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f"dist_mx_{abbrvs_key}"))
        plt.show()

        fig = plt.figure()
        sns.histplot(data=pd.DataFrame(dist_list), x=0, bins=30)
        fig.savefig(os.path.join(self.output_path, f"dist_histogram"))
