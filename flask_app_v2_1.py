"""
Task Two: Gene Homologs
Created by: Andy Martin

Current Version: v2.1.0

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

# Conditions of Use pageage
@app.route('/PPERisk/COU')
def conditions_of_use():
    """Render the Conditions of Use page."""
    return render_template('COU.html')

# Frequently Asked Questions page
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


def calculate_mechanism_total(mechanism_stages, control_counts):
    """
    Helper to calculate average mechanism efficiency across stages.

    Args:
        mechanism_stages: List of efficiency values for each stage
        control_counts: List of control particle counts for each stage

    Returns:
        Formatted percentage string
    """
    stage_means = []
    for stage_eff, control in zip(mechanism_stages, control_counts):
        capped = np.minimum(stage_eff / 100, 1 - (1 / control))
        stage_means.append(np.mean(capped))
    return f"{round(100 * np.mean(stage_means), 2)}%"


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
    material_params = [
        {'name': 'fiber_diameter', 'loc': df, 'scale': dfsd, 'min': 0.3587, 'max': 15},
        {'name': 'packing_density', 'loc': z, 'scale': zsd, 'min': 0.01, 'max': 1},
        {'name': 'thickness', 'loc': L, 'scale': Lsd, 'min': 150, 'max': 2000}
    ]

    # Generate truncated normal distributions
    material_values = {}
    for param in material_params:
        # Calculate truncation parameters for truncnorm
        a = (param['min'] - param['loc']) / param['scale']
        b = (param['max'] - param['loc']) / param['scale']

        # Generate distribution
        material_values[param['name']] = pd.Series(
            truncnorm.rvs(a=a, b=b, loc=param['loc'], scale=param['scale'], size=num_simulations)
        )

    # Extract values for clarity
    var1_values = material_values['fiber_diameter']
    var2_values = material_values['packing_density']
    var3_values = material_values['thickness']

    # Define particle size ranges for each stage
    stage_config = [
        {'name': '7.0+', 'min': 7.0, 'max': 10, 'control': 5080},
        {'name': '4.7-7.0', 'min': 4.7, 'max': 7.0, 'control': 4820},
        {'name': '3.3-4.7', 'min': 3.3, 'max': 4.7, 'control': 6740},
        {'name': '2.2-3.3', 'min': 2.2, 'max': 3.3, 'control': 9596},
        {'name': '1.1-2.2', 'min': 1.1, 'max': 2.2, 'control': 19886},
        {'name': '0.65-1.1', 'min': 0.65, 'max': 1.1, 'control': 26824}
    ]

    # Generate particle size distributions and calculate efficiencies for each stage
    stage_results = []
    for stage in stage_config:
        # Generate uniform distribution for this stage's particle sizes
        particle_sizes = np.random.uniform(stage['min'], stage['max'], size=num_simulations)

        # Calculate filtration efficiency
        overall, interception, impaction, diffusion, diff_int = calculate_percent_ofe(
            var1_values, particle_sizes, var2_values, var3_values
        )

        stage_results.append({
            'name': stage['name'],
            'control': stage['control'],
            'overall': overall,
            'interception': interception,
            'impaction': impaction,
            'diffusion': diffusion,
            'diffusion_interception': diff_int
        })

    # Extract control counts for all stages
    control_counts = [stage['control'] for stage in stage_results]

    # Calculate total efficiencies for each mechanism using loops
    mechanisms = ['interception', 'impaction', 'diffusion', 'diffusion_interception', 'overall']
    mechanism_totals = {}

    for mechanism in mechanisms:
        mechanism_values = [stage[mechanism] for stage in stage_results]
        mechanism_totals[mechanism] = calculate_mechanism_total(mechanism_values, control_counts)

    # Calculate log reduction and normalized values for each stage
    normalized_stages = []
    for stage in stage_results:
        # Calculate log reduction
        log_reduction = calculate_log_reduction(stage['overall'], stage['control'])

        # Normalize by dividing by the -log10 of the limit of detection
        normalized = log_reduction / -np.log10(1 / stage['control'])
        normalized_stages.append(normalized)

    percentiles_dict = calculate_percentiles(normalized_stages)

    # Prepare result dictionaries
    results1 = {
        "Parameters": ["Fiber Diameter (µm)", "Fiber Diameter SD", "Packing Density (α)",
                       "Packing Density SD", "Thickness (µm)", "Thickness SD"],
        "Input": [df, dfsd, z, zsd, L, Lsd],
    }

    results2 = percentiles_dict

    results3 = {
        "diffusion": [mechanism_totals['diffusion']],
        "interception": [mechanism_totals['interception']],
        "impaction": [mechanism_totals['impaction']],
        "diffusion interception": [mechanism_totals['diffusion_interception']]
    }

    results4 = {
        "overall filter efficiency": [mechanism_totals['overall']]
    }

    results5 = {
        "y": normalized_stages,
        "x": [stage['name'] for stage in stage_results]
    }

    return results1, results2, results3, results4, results5


if __name__ == '__main__':
    app.run()
