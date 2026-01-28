"""
Flask application for mask filtration efficiency analysis.
Provides Monte Carlo simulation for respirator performance evaluation.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy.stats import mannwhitneyu, truncnorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba


# Configuration
class Config:
    DEBUG = True
    RESPIRATOR_DATA_PATH = 'PPE-Calculator/respiratordf.csv'


# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Load reference data
respirator_df = pd.read_csv(Config.RESPIRATOR_DATA_PATH)


# Routes
@app.route('/')
def masks():
    """Render the main mask calculator page."""
    return render_template('masks.html')


@app.route('/', methods=['POST'])
def mask_calc():
    """
    Handle mask calculation request.
    Processes user input parameters and returns filtration efficiency analysis.
    """
    if request.method != 'POST':
        return jsonify({'error': 'Invalid request method'}), 405

    # Extract form data
    params = extract_form_parameters(request.form)

    # Perform Monte Carlo simulation
    _, percentiles, mechanisms, ofe, graph_data = monte_carlo(
        params['df'], params['dfsd'],
        params['z'], params['zsd'],
        params['thick'], params['thicksd']
    )

    # Generate results
    results = {
        'result2': generate_html_table(percentiles),
        'result3': generate_html_table(mechanisms),
        'result4': generate_html_table(ofe),
        'graph': create_comparison_graph(graph_data, respirator_df)
    }

    return jsonify(results)


@app.route('/PPERisk/COU')
def conditions_of_use():
    """Render the Conditions of Use page."""
    return render_template('COU.html')


@app.route('/PPERisk/FAQ')
def faq():
    """Render the FAQ page."""
    return render_template('FAQ.html')


# Helper Functions
def extract_form_parameters(form_data):
    """
    Extract and validate form parameters.

    Args:
        form_data: Flask request form data

    Returns:
        Dictionary of validated parameters
    """
    return {
        'df': form_data.get('df', type=float),
        'dfsd': form_data.get('dfsd', type=float),
        'thick': form_data.get('thick', type=float),
        'thicksd': form_data.get('thicksd', type=float),
        'z': form_data.get('packingdensity', type=float),
        'zsd': form_data.get('packingdensitysd', type=float)
    }


def calculate_percentiles(normalized_log_stages):
    """
    Calculate percentiles for all stages.

    Args:
        normalized_log_stages: List of normalized log reduction values for each stage

    Returns:
        Dictionary containing percentile data for all stages
    """
    percentiles = [5, 25, 50, 75, 95]
    stage_names = ["1", "2", "3", "4", "5", "6"]
    particle_ranges = [
        "7.0+", "4.7-7.0", "3.3-4.7",
        "2.2-3.3", "1.1-2.2", "0.65-1.1"
    ]

    results = {
        "Stage": stage_names,
        "Particle Size Range (µm)": particle_ranges
    }

    for p in percentiles:
        column_name = f"Normalized Log Reduction Value {p}th percentile"
        results[column_name] = [
            np.percentile(stage, p) for stage in normalized_log_stages
        ]

    return results


def calculate_log_reduction(efficiency, control_count):
    """
    Calculate log10 reduction from efficiency percentage.

    Args:
        efficiency: Filtration efficiency (0-100)
        control_count: Control particle count for normalization

    Returns:
        Log10 reduction value
    """
    return -np.log10(np.maximum(1 - (efficiency / 100), 1 / control_count))


def create_comparison_graph(user_data, reference_data):
    """
    Create violin plot comparing user material to N95 reference.

    Args:
        user_data: Dictionary with 'x' and 'y' keys for user material data
        reference_data: DataFrame with reference respirator data

    Returns:
        Base64 encoded PNG image string
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Prepare user data
    user_df = pd.DataFrame(user_data)
    user_df = user_df.explode('y')
    user_df['y'] = user_df['y'].astype('float')
    user_df['id'] = "Your material"

    # Prepare reference data
    ref_df = reference_data.copy()
    ref_df['id'] = "n95 model reference"

    # Combine datasets
    combined_df = pd.concat([user_df, ref_df])
    combined_df = combined_df.explode('y')
    combined_df['y'] = combined_df['y'].astype('float')

    # Create violin plot
    sns.set_theme(style='whitegrid')
    ax.yaxis.grid(which='both', alpha=0.2, visible=True)

    colors = {
        'Your material': 'lemonchiffon',
        'n95 model reference': 'lightsteelblue'
    }

    sns.violinplot(
        ax=ax, x='x', y='y', data=combined_df,
        palette=colors, density_norm='count', hue='id',
        split=True, gap=0.2, linewidth=1.2, cut=0, inner='quart'
    )

    # Labels and legend
    plt.xlabel("Particle Size Range (µm)")
    plt.ylabel("Normalized Log10 Reduction")

    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--'),
        Line2D([0], [0], color='black', linestyle='--'),
        Patch(facecolor='lightsteelblue', edgecolor='black'),
        Patch(facecolor='lightcoral', edgecolor='black', alpha=0.5),
        Patch(facecolor='darkseagreen', edgecolor='black', alpha=0.5),
    ]

    plt.legend(
        legend_elements,
        [
            'Median',
            '25th & 75th Percentile',
            'N95 reference respirator',
            'Significantly less\nthan reference,\np-value < 0.05',
            'Not significantly less\nthan reference,\np-value > 0.05'
        ],
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )

    # Apply statistical coloring
    apply_statistical_coloring(ax, user_df, ref_df, combined_df)

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_str


def apply_statistical_coloring(ax, user_df, ref_df, combined_df):
    """
    Apply color coding based on Mann-Whitney U test results.

    Args:
        ax: Matplotlib axis object
        user_df: User material DataFrame
        ref_df: Reference respirator DataFrame
        combined_df: Combined DataFrame
    """
    user_facecolor = np.array([0.9754901960784312, 0.9607843137254901,
                                0.8284313725490198, 1.0])

    plot_count = 0
    stage_count = 0
    line_count = 0

    # Color median lines red
    for line in ax.lines:
        if line.get_linestyle() == '--':
            plot_count += 1
            if plot_count % 3 == 2:
                line.set_color('red')
                line_count += 1
        elif line.get_linestyle() == '-':
            line_count += 1

        if plot_count == 2:
            stage_count += 1
            plot_count = 0

        # Apply statistical test coloring
        if line_count > 0 and line_count <= len(ax.collections):
            facecolor = ax.collections[line_count - 1].get_facecolor()

            if np.array_equal(facecolor[0], user_facecolor):
                stage_x = combined_df['x'].unique()[stage_count]
                user_values = user_df['y'][user_df['x'] == stage_x]
                ref_values = ref_df['y'][ref_df['x'] == stage_x]

                _, p_value = mannwhitneyu(user_values, ref_values, alternative='less')

                if p_value < 0.05:
                    color = to_rgba('lightcoral', alpha=0.5)
                else:
                    color = to_rgba('darkseagreen', alpha=0.5)

                ax.collections[line_count - 1].set_facecolor(color)


def generate_html_table(data_dict):
    """
    Convert dictionary to HTML table.

    Args:
        data_dict: Dictionary with table data

    Returns:
        HTML table string
    """
    df = pd.DataFrame(data_dict)
    return df.to_html(index=False, justify="center")


def calculate_percent_ofe(df, dp, z, L):
    """
    Calculate Overall Filtration Efficiency (OFE) and individual mechanisms.

    Uses classical filtration theory to calculate efficiency based on:
    - Diffusion
    - Interception
    - Impaction
    - Diffusion-Interception coupling

    Args:
        df: Fiber diameter (µm)
        dp: Particle diameter (µm)
        z: Packing density (α)
        L: Thickness (µm)

    Returns:
        Tuple of (overall_efficiency, interception, impaction, diffusion, diffusion_interception)
        All values as percentages (0-100)
    """
    fiber_diameter = df
    particle_diameter = dp
    packing_density = z
    thickness = L
    face_velocity = 0.66727

    # Dimensionless parameters
    R = particle_diameter / fiber_diameter
    kuwabara = -0.5 * np.log(packing_density) - 0.75 + packing_density - 0.25 * packing_density**2
    knudsen = 2 * 0.067 / fiber_diameter

    # Cunningham slip correction factors
    Cu = 1 + knudsen * (1.257 + 0.4 * np.exp(-1.1 / knudsen))
    Cn = 1 + (2 * 0.067 / particle_diameter) * (1.257 + 0.4 * np.exp(-1.1 / (2 * 0.067 / particle_diameter)))

    # Diffusion coefficient
    D = (307.15 * Cn * (1.38054 * 10**-23)) / (3 * np.pi * 1.9e-5 * (particle_diameter / 1e6))

    # Peclet and Stokes numbers
    peclet = (fiber_diameter / 1000000) * face_velocity / D
    Stk = (Cu * 993 * (particle_diameter / 1e6)**2 * face_velocity) / (18 * 1.9e-5 * (fiber_diameter / 1e6))

    # Mechanism efficiencies
    Liu = 1.6 * (np.cbrt((1 - packing_density) / kuwabara)) * (peclet**(-2 / 3)) * (1 + 0.388 * knudsen * (np.cbrt(((1 - packing_density) * peclet) / kuwabara)))
    cD = 1 + 0.388 * knudsen * (np.cbrt(((1 - packing_density) * peclet) / kuwabara))

    interception = 0.6 * (1 + ((1.996 * knudsen) / R)) * ((1 - packing_density) / kuwabara) * ((R**2) / (1 + R))
    impaction = 0.0334 * (Stk**(3 / 2))
    diffusion = 1.6 * (np.cbrt((1 - packing_density) / kuwabara)) * (peclet**(-2 / 3)) * (cD / (1 + Liu))

    Ks = -0.5 * np.log(packing_density) - 0.52 + 0.64 * packing_density + 1.43 * (1 - packing_density) * knudsen
    diffusion_interception = 1.24 * (R**(2 / 3) / np.sqrt(Ks * peclet))

    # Overall efficiency
    overall_efficiency = interception + impaction + diffusion + diffusion_interception

    # Filtration efficiencies
    base_exp = (-2 * packing_density * thickness) / (np.pi * (fiber_diameter / 2) * (1 - packing_density))

    overall_fe = 1 - np.exp(overall_efficiency * base_exp)
    interception_fe = 1 - np.exp(interception * base_exp)
    impaction_fe = 1 - np.exp(impaction * base_exp)
    diffusion_fe = 1 - np.exp(diffusion * base_exp)
    diffusion_interception_fe = 1 - np.exp(diffusion_interception * base_exp)

    # Convert to percentages
    return (100 * overall_fe, 100 * interception_fe, 100 * impaction_fe,
            100 * diffusion_fe, 100 * diffusion_interception_fe)


def monte_carlo(df, dfsd, z, zsd, L, Lsd):
    """
    Perform Monte Carlo simulation for filtration efficiency.

    Runs 10,000 simulations with truncated normal distributions for material
    properties and uniform distributions for particle sizes across 6 stages.

    Args:
        df: Fiber diameter mean (µm)
        dfsd: Fiber diameter standard deviation (µm)
        z: Packing density mean (α)
        zsd: Packing density standard deviation
        L: Thickness mean (µm)
        Lsd: Thickness standard deviation (µm)

    Returns:
        Tuple of 5 dictionaries:
        - results1: Input parameters
        - results2: Percentile data by stage
        - results3: Mechanism contributions
        - results4: Overall filter efficiency
        - results5: Graph data
    """
    num_simulations = 10000
    np.random.seed(1234)

    # Define truncation bounds and distribution parameters
    loc1, scale1 = df, dfsd
    loc2, scale2 = z, zsd
    loc3, scale3 = L, Lsd

    a_var1, b_var1 = 0.3587, 15
    a_var2, b_var2 = 0.01, 1
    a_var3, b_var3 = 150, 2000

    # Calculate truncation parameters for truncnorm
    a1, b1 = (a_var1 - loc1) / scale1, (b_var1 - loc1) / scale1
    a2, b2 = (a_var2 - loc2) / scale2, (b_var2 - loc2) / scale2
    a3, b3 = (a_var3 - loc3) / scale3, (b_var3 - loc3) / scale3

    # Generate truncated normal distributions
    var1_values = pd.Series(truncnorm.rvs(a=a1, b=b1, loc=loc1, scale=scale1, size=num_simulations))
    var2_values = pd.Series(truncnorm.rvs(a=a2, b=b2, loc=loc2, scale=scale2, size=num_simulations))
    var3_values = pd.Series(truncnorm.rvs(a=a3, b=b3, loc=loc3, scale=scale3, size=num_simulations))

    # Generate uniform distributions for particle sizes at each stage
    var4_values_stage1 = np.random.uniform(7.0, 10, size=num_simulations)
    var4_values_stage2 = np.random.uniform(4.7, 7.0, size=num_simulations)
    var4_values_stage3 = np.random.uniform(3.3, 4.7, size=num_simulations)
    var4_values_stage4 = np.random.uniform(2.2, 3.3, size=num_simulations)
    var4_values_stage5 = np.random.uniform(1.1, 2.2, size=num_simulations)
    var4_values_stage6 = np.random.uniform(0.65, 1.1, size=num_simulations)

    # Calculate filtration efficiency for each stage
    results_stage1, int1, imp1, dif1, intdif1 = calculate_percent_ofe(var1_values, var4_values_stage1, var2_values, var3_values)
    results_stage2, int2, imp2, dif2, intdif2 = calculate_percent_ofe(var1_values, var4_values_stage2, var2_values, var3_values)
    results_stage3, int3, imp3, dif3, intdif3 = calculate_percent_ofe(var1_values, var4_values_stage3, var2_values, var3_values)
    results_stage4, int4, imp4, dif4, intdif4 = calculate_percent_ofe(var1_values, var4_values_stage4, var2_values, var3_values)
    results_stage5, int5, imp5, dif5, intdif5 = calculate_percent_ofe(var1_values, var4_values_stage5, var2_values, var3_values)
    results_stage6, int6, imp6, dif6, intdif6 = calculate_percent_ofe(var1_values, var4_values_stage6, var2_values, var3_values)

    # Control counts for each stage
    control_counts = [5080, 5080, 4820, 6740, 9596, 26824]

    # Calculate total efficiencies for each mechanism
    def calculate_mechanism_total(mechanism_stages, control_counts):
        """Helper to calculate average mechanism efficiency across stages."""
        stage_means = []
        for stage_eff, control in zip(mechanism_stages, control_counts):
            capped = np.minimum(stage_eff / 100, 1 - (1 / control))
            stage_means.append(np.mean(capped))
        return f"{round(100 * np.mean(stage_means), 2)}%"

    intercept_total = calculate_mechanism_total([int1, int2, int3, int4, int5, int6], control_counts)
    impaction_total = calculate_mechanism_total([imp1, imp2, imp3, imp4, imp5, imp6], control_counts)
    diffusion_total = calculate_mechanism_total([dif1, dif2, dif3, dif4, dif5, dif6], control_counts)
    interception_diffusion_total = calculate_mechanism_total([intdif1, intdif2, intdif3, intdif4, intdif5, intdif6], control_counts)
    total_total = calculate_mechanism_total([results_stage1, results_stage2, results_stage3, results_stage4, results_stage5, results_stage6], control_counts)

    # Calculate log reduction values
    logresults_stage1 = calculate_log_reduction(results_stage1, 5080)
    logresults_stage2 = calculate_log_reduction(results_stage2, 4820)
    logresults_stage3 = calculate_log_reduction(results_stage3, 6740)
    logresults_stage4 = calculate_log_reduction(results_stage4, 9596)
    logresults_stage5 = calculate_log_reduction(results_stage5, 19886)
    logresults_stage6 = calculate_log_reduction(results_stage6, 26824)

    # Normalize log reduction values
    normalizedlog_stage1 = logresults_stage1 / -np.log10(1 / 5080)
    normalizedlog_stage2 = logresults_stage2 / -np.log10(1 / 4820)
    normalizedlog_stage3 = logresults_stage3 / -np.log10(1 / 6740)
    normalizedlog_stage4 = logresults_stage4 / -np.log10(1 / 9596)
    normalizedlog_stage5 = logresults_stage5 / -np.log10(1 / 19886)
    normalizedlog_stage6 = logresults_stage6 / -np.log10(1 / 26824)

    # Calculate percentiles for all stages
    normalized_stages = [normalizedlog_stage1, normalizedlog_stage2, normalizedlog_stage3,
                         normalizedlog_stage4, normalizedlog_stage5, normalizedlog_stage6]

    percentiles_dict = calculate_percentiles(normalized_stages)

    # Prepare result dictionaries
    results1 = {
        "Parameters": ["Fiber Diameter (µm)", "Fiber Diameter SD", "Packing Density (α)",
                       "Packing Density SD", "Thickness (µm)", "Thickness SD"],
        "Input": [df, dfsd, z, zsd, L, Lsd],
    }

    results2 = percentiles_dict

    results3 = {
        "diffusion": [diffusion_total],
        "interception": [intercept_total],
        "impaction": [impaction_total],
        "diffusion interception": [interception_diffusion_total]
    }

    results4 = {
        "overall filter efficiency": [total_total]
    }

    results5 = {
        "y": normalized_stages,
        "x": ["7.0+", "4.7-7.0", "3.3-4.7", "2.2-3.3", "1.1-2.2", "0.65-1.1"]
    }

    return results1, results2, results3, results4, results5


if __name__ == '__main__':
    app.run()
    app.run()