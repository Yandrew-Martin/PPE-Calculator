from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from statistics import mean
import seaborn
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import matplotlib
from numpy.core.numeric import array_equal
from sqlalchemy.dialects.postgresql import array

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import truncnorm, ttest_ind, mannwhitneyu
from scipy.special import cbrt

#pythonanywhere is the directory name for my test environment - replace with correct directory name
respiratordf = pd.read_csv('respiratordf.csv')

app = Flask(__name__)
app.config["DEBUG"] = False

#@app.route('/')
#def home():
#    return render_template('home.html')

@app.route('/')
def masks():
    return render_template('masks.html')

@app.route('/', methods=['POST'])
def maskcalc():

    if request.method == 'POST':
        df = request.form.get("df",type=float)
        dfsd = request.form.get("dfsd",type=float)
        thick = request.form.get("thick",type=float)
        thicksd = request.form.get("thicksd",type=float)
        #packing density
        z = request.form.get("packingdensity",type=float)
        zsd = request.form.get("packingdensitysd",type=float)

        #user input, percentiles, mechanisms, OFE, output for graph
        result1, result2, result3, result4, result5 = monte_carlo(df,dfsd,z,zsd,thick,thicksd)
        #make tables
        #printresults = printResults(result1)
        printresults2 = printResults(result2)
        printresults3 = printResults(result3)
        printresults4 = printResults(result4)
        img_str = makeGraph(result5, respiratordf)

        return jsonify({
            #'result': printresults,
            'result2': printresults2,
            'result3': printresults3,
            'result4': printresults4,
            'graph': img_str
        })

@app.route('/PPERisk/COU')
def cou():
    return render_template('COU.html')

@app.route('/PPERisk/FAQ')
def faq():
    return render_template('FAQ.html')

def calculate_H24(df, dp, z, L):
    B1 = df
    B2 = dp
    B4 = z
    B5 = L
    B6 = 0.66727
    B8 = B2 / B1
    B9 = -0.5 * np.log(B4) - 0.75 + B4 - 0.25 * B4**2
    B14 = 2 * 0.067 / B1
    B15 = 1 + B14 * (1.257 + 0.4 * np.exp(-1.1 / B14))
    B16 = 1 + (2 * 0.067 / B2) * (1.257 + 0.4 * np.exp(-1.1 / (2 * 0.067 / B2)))
    B12 = (307.15 * B16 * (1.38054 * 10**-23)) / (3 * np.pi * 1.9e-5 * (B2 / 1e6))
    B11 = (B1 / 1000000) * B6 / B12
    B13 = (B15 * 993 * (B2 / 1e6)**2 * B6) / (18 * 1.9e-5 * (B1 / 1e6))
    B17 = 1.6 * (cbrt((1 - B4) / B9)) * (B11**(-2 / 3)) * (1 + 0.388 * B14 * (cbrt(((1 - B4) * B11) / B9)))
    B18 = 1 + 0.388 * B14 * (cbrt(((1 - B4) * B11) / B9))
    B20 = 0.6 * (1 + ((1.996 * B14) / B8)) * ((1 - B4) / B9) * ((B8**2) / (1 + B8))
    B21 = 0.0334 * (B13**(3 / 2))
    B22 = 1.6 * (cbrt((1 - B4) / B9)) * (B11**(-2 / 3)) * (B18 / (1 + B17))
    Ks = -0.5 * np.log(B4) - 0.52 + 0.64 * B4 + 1.43 * (1 - B4) * B14
    B25 = 1.24 * (B8**(2 / 3) / np.sqrt(Ks * B11))
    B24 = B20 + B21 + B22 + B25
    D24 = np.exp((-2 * B4 * B24 * B5) / (np.pi * (B1 / 2) * (1 - B4)))
    D20 = np.exp((-2 * B4 * B20 * B5) / (np.pi * (B1 / 2) * (1 - B4)))
    D21 = np.exp((-2 * B4 * B21 * B5) / (np.pi * (B1 / 2) * (1 - B4)))
    D22 = np.exp((-2 * B4 * B22 * B5) / (np.pi * (B1 / 2) * (1 - B4)))
    D25 = np.exp((-2 * B4 * B25 * B5) / (np.pi * (B1 / 2) * (1 - B4)))
    F24 = 100 * D24
    F20 = 100 * D20
    F21 = 100 * D21
    F22 = 100 * D22
    F25 = 100 * D25
    H24 = 100 - F24
    H20 = 100 - F20
    H21 = 100 - F21
    H22 = 100 - F22
    H25 = 100 - F25
    return H24, H20, H21, H22, H25

def monte_carlo(df, dfsd, z, zsd, L, Lsd):
    num_simulations = 10000
    np.random.seed(1234)

    #min/max parameters and mean/sd parameters
    loc1, scale1 = df, dfsd
    loc2, scale2 = z, zsd
    loc3, scale3 = L, Lsd
    a_var1, b_var1 = 0.3587, 15
    a_var2, b_var2 = 0.01, 1
    a_var3, b_var3 = 150, 2000
    #required calculation for truncnorm
    a1, b1 = (a_var1 - loc1) / scale1, (b_var1 - loc1) / scale1
    a2, b2 = (a_var2 - loc2) / scale2, (b_var2 - loc2) / scale2
    a3, b3 = (a_var3 - loc3) / scale3, (b_var3 - loc3) / scale3

    #truncnorm distributions
    var1_values = pd.Series(truncnorm.rvs(a=a1, b=b1, loc=loc1, scale=scale1, size=10000))
    var2_values = pd.Series(truncnorm.rvs(a=a2, b=b2, loc=loc2, scale=scale2, size=10000))
    var3_values = pd.Series(truncnorm.rvs(a=a3, b=b3, loc=loc3, scale=scale3, size=10000))

    #uniform random values for different stages
    var4_values_stage1 = np.random.uniform(7.0, 10, size=num_simulations)
    var4_values_stage2 = np.random.uniform(4.7, 7.0, size=num_simulations)
    var4_values_stage3 = np.random.uniform(3.3, 4.7, size=num_simulations)
    var4_values_stage4 = np.random.uniform(2.2, 3.3, size=num_simulations)
    var4_values_stage5 = np.random.uniform(1.1, 2.2, size=num_simulations)
    var4_values_stage6 = np.random.uniform(0.65, 1.1, size=num_simulations)

    #calculations at each stage for each mechanism and total
    results_stage1,int1,imp1,dif1,intdif1 = calculate_H24(var1_values, var4_values_stage1, var2_values, var3_values)
    results_stage2,int2,imp2,dif2,intdif2 = calculate_H24(var1_values, var4_values_stage2, var2_values, var3_values)
    results_stage3,int3,imp3,dif3,intdif3 = calculate_H24(var1_values, var4_values_stage3, var2_values, var3_values)
    results_stage4,int4,imp4,dif4,intdif4 = calculate_H24(var1_values, var4_values_stage4, var2_values, var3_values)
    results_stage5,int5,imp5,dif5,intdif5 = calculate_H24(var1_values, var4_values_stage5, var2_values, var3_values)
    results_stage6,int6,imp6,dif6,intdif6 = calculate_H24(var1_values, var4_values_stage6, var2_values, var3_values)
    intercept_total = str(round(100*mean((mean(np.minimum((int1/100),1-(1/5080))),mean(np.minimum((int2/100),1-(1/5080))),mean(np.minimum((int3/100),1-(1/4820))),mean(np.minimum((int4/100),1-(1/6740))),mean(np.minimum((int5/100),1-(1/9596))),mean(np.minimum((int6/100),1-(1/26824))))),2)) + "%"
    impaction_total = str(round(100*mean((mean(np.minimum((imp1/100),1-(1/5080))),mean(np.minimum((imp2/100),1-(1/5080))),mean(np.minimum((imp3/100),1-(1/4820))),mean(np.minimum((imp4/100),1-(1/6740))),mean(np.minimum((imp5/100),1-(1/9596))),mean(np.minimum((imp6/100),1-(1/26824))))),2)) + "%"
    diffusion_total = str(round(100*mean((mean(np.minimum((dif1/100),1-(1/5080))),mean(np.minimum((dif2/100),1-(1/5080))),mean(np.minimum((dif3/100),1-(1/4820))),mean(np.minimum((dif4/100),1-(1/6740))),mean(np.minimum((dif5/100),1-(1/9596))),mean(np.minimum((dif6/100),1-(1/26824))))),2)) + "%"
    interception_diffusion_total = str(round(100*mean((mean(np.minimum((intdif1/100),1-(1/5080))),mean(np.minimum((intdif2/100),1-(1/5080))),mean(np.minimum((intdif3/100),1-(1/4820))),mean(np.minimum((intdif4/100),1-(1/6740))),mean(np.minimum((intdif5/100),1-(1/9596))),mean(np.minimum((intdif6/100),1-(1/26824))))),2)) + "%"
    total_total = str(round(100*mean((mean(np.minimum((results_stage1/100),1-(1/5080))),mean(np.minimum((results_stage2/100),1-(1/5080))),mean(np.minimum((results_stage3/100),1-(1/4820))),mean(np.minimum((results_stage4/100),1-(1/6740))),mean(np.minimum((results_stage5/100),1-(1/9596))),mean(np.minimum((results_stage6/100),1-(1/26824))))),2)) + "%"
    #log calculations
    logresults_stage1 = logresults_stage(results_stage1, 5080)
    logresults_stage2 = logresults_stage(results_stage2, 4820)
    logresults_stage3 = logresults_stage(results_stage3, 6740)
    logresults_stage4 = logresults_stage(results_stage4, 9596)
    logresults_stage5 = logresults_stage(results_stage5, 19886)
    logresults_stage6 = logresults_stage(results_stage6, 26824)
    #normalized by dividing by the -log10 of the limit of detection
    normalizedlog_stage1 = logresults_stage1 / -np.log10(1 / 5080)
    normalizedlog_stage2 = logresults_stage2 / -np.log10(1 / 4820)
    normalizedlog_stage3 = logresults_stage3 / -np.log10(1 / 6740)
    normalizedlog_stage4 = logresults_stage4 / -np.log10(1 / 9596)
    normalizedlog_stage5 = logresults_stage5 / -np.log10(1 / 19886)
    normalizedlog_stage6 = logresults_stage6 / -np.log10(1 / 26824)
    fifthpercentile1 = np.percentile(normalizedlog_stage1,5)
    fifthpercentile2 = np.percentile(normalizedlog_stage2,5)
    fifthpercentile3 = np.percentile(normalizedlog_stage3,5)
    fifthpercentile4 = np.percentile(normalizedlog_stage4,5)
    fifthpercentile5 = np.percentile(normalizedlog_stage5,5)
    fifthpercentile6 = np.percentile(normalizedlog_stage6,5)
    ninefivepercentile1 = np.percentile(normalizedlog_stage1,95)
    ninefivepercentile2 = np.percentile(normalizedlog_stage2,95)
    ninefivepercentile3 = np.percentile(normalizedlog_stage3,95)
    ninefivepercentile4 = np.percentile(normalizedlog_stage4,95)
    ninefivepercentile5 = np.percentile(normalizedlog_stage5,95)
    ninefivepercentile6 = np.percentile(normalizedlog_stage6,95)
    twofivepercentile1 = np.percentile(normalizedlog_stage1,25)
    twofivepercentile2 = np.percentile(normalizedlog_stage2,25)
    twofivepercentile3 = np.percentile(normalizedlog_stage3,25)
    twofivepercentile4 = np.percentile(normalizedlog_stage4,25)
    twofivepercentile5 = np.percentile(normalizedlog_stage5,25)
    twofivepercentile6 = np.percentile(normalizedlog_stage6,25)
    fiftypercentile1 = np.percentile(normalizedlog_stage1,50)
    fiftypercentile2 = np.percentile(normalizedlog_stage2,50)
    fiftypercentile3 = np.percentile(normalizedlog_stage3,50)
    fiftypercentile4 = np.percentile(normalizedlog_stage4,50)
    fiftypercentile5 = np.percentile(normalizedlog_stage5,50)
    fiftypercentile6 = np.percentile(normalizedlog_stage6,50)
    sevenfivepercentile1 = np.percentile(normalizedlog_stage1,75)
    sevenfivepercentile2 = np.percentile(normalizedlog_stage2,75)
    sevenfivepercentile3 = np.percentile(normalizedlog_stage3,75)
    sevenfivepercentile4 = np.percentile(normalizedlog_stage4,75)
    sevenfivepercentile5 = np.percentile(normalizedlog_stage5,75)
    sevenfivepercentile6 = np.percentile(normalizedlog_stage6,75)

    results1 = {
        "Parameters": ["Fiber Diameter (µm)","Fiber Diameter SD","Packing Density (α)","Packing Density SD","Thickness (µm)","Thickness SD"],
        "Input":[df,dfsd,z,zsd,L,Lsd],
        }
    results2 = {
        "Stage": ["1","2","3","4","5","6"],
        "Particle Size Range (µm)": ["7.0+","4.7-7.0","3.3-4.7","2.2-3.3","1.1-2.2","0.65-1.1"],
        "Normalized Log Reduction Value 5th percentile": [fifthpercentile1,fifthpercentile2,fifthpercentile3,fifthpercentile4,fifthpercentile5,fifthpercentile6],
        "Normalized Log Reduction Value 25th percentile": [twofivepercentile1,twofivepercentile2,twofivepercentile3,twofivepercentile4,twofivepercentile5,twofivepercentile6],
        "Normalized Log Reduction Value 50th percentile": [fiftypercentile1,fiftypercentile2,fiftypercentile3,fiftypercentile4,fiftypercentile5,fiftypercentile6],
        "Normalized Log Reduction Value 75th percentile": [sevenfivepercentile1,sevenfivepercentile2,sevenfivepercentile3,sevenfivepercentile4,sevenfivepercentile5,sevenfivepercentile6],
        "Normalized Log Reduction Value 95th percentile": [ninefivepercentile1,ninefivepercentile2,ninefivepercentile3,ninefivepercentile4,ninefivepercentile5,ninefivepercentile6]
        }
    results3 = {
        "diffusion":[diffusion_total],
        "interception":[intercept_total],
        "impaction":[impaction_total],
        "diffusion interception":[interception_diffusion_total]
        }
    results4 = {
        "overall filter efficiency":[total_total]
        }
    results5 = {
        "y":[normalizedlog_stage1,normalizedlog_stage2,normalizedlog_stage3,normalizedlog_stage4,normalizedlog_stage5,normalizedlog_stage6],
        "x":["7.0+","4.7-7.0","3.3-4.7","2.2-3.3","1.1-2.2","0.65-1.1"]
        }
    return results1, results2, results3, results4, results5

#returns log10 reduction
def logresults_stage(percentile, control):
    return -np.log10(np.maximum(1 - (percentile / 100), 1 / control))

def makeGraph(results, respirator):

    fig, ax = plt.subplots(figsize=(8,4))

    df1 = pd.DataFrame(results)
    df1 = df1.explode('y')
    df1['y'] = df1['y'].astype('float')
    df1['id'] = "Your material"
    df2 = respirator
    df2['id'] = "n95 model reference"
    df3 = pd.concat([df1, df2])
    df3 = df3.explode('y')
    df3['y'] = df3['y'].astype('float')
    seaborn.set_theme(style = 'whitegrid')
    ax.yaxis.grid(which='both',alpha=0.2,visible=True)
    colors = {'Your material':'lemonchiffon','n95 model reference':'lightsteelblue'}
    seaborn.violinplot(ax=ax, x ='x', y ='y', data = df3,palette=colors,density_norm='count', hue='id', split=True,gap=.2,linewidth=1.2,cut=0, inner='quart')
    plt.xlabel("Particle Size Range (µm)")
    plt.ylabel("Normalized Log10 Reduction")
    legend_elements = [Line2D([0], [0], color='red', linestyle='--'),
                       Line2D([0], [0], color='black', linestyle='--'),
                       Patch(facecolor='lightsteelblue', edgecolor='black'),
                       Patch(facecolor='lightcoral', edgecolor='black', alpha=0.5),
                       Patch(facecolor='darkseagreen', edgecolor='black', alpha=0.5),
                        ]
    plt.legend(legend_elements,['Median','25th & 75th Percentile','n95 reference respirator','Significantly less\n than reference,\n p-value < 0.05','Not significantly less\n than reference,\n p-value > 0.05'],loc='center left', bbox_to_anchor=(1, 0.5))

    #pastel blue
    #user_facecolor = np.array([0.6720588235294118, 0.789705882352941, 0.9161764705882354, 1.0])
    #lemonchiffon
    user_facecolor = np.array([0.9754901960784312, 0.9607843137254901, 0.8284313725490198, 1.0])

    #parameters for next for loop
    count=0
    plotcount=0
    sigcount=0
    linecount=0

    #my way of only turning the median lines red. must be used because when the violin plot is a flat line, it counts as one line object for one plot; however, when the violin plot exists, there are three line objects for one plot. This changes depending on how many stages are at 100% log10 reduction.
    #the list of line2D objects also includes the reference plot lines, so i created this counting for loop to keep track of true median lines.
    for l in ax.lines:
        #when the plot is generated and has three lines. middle is median
        if l.get_linestyle()=='--':
            count+=1
            if count%3==2:
                l.set_color('red')
                plotcount+=1
                linecount+=1
        #when the plot is just a line
        if l.get_linestyle()=='-':
            plotcount+=1
        #counts each plot pair; value is used for t test
        if plotcount==2:
            sigcount+=1
            plotcount=0
        #check if we are observing the user side of the plot to change the facecolor to red or green
        facecolor = ax.collections[linecount-1].get_facecolor()
        #t-test to determine face color of the violin plot
        if np.array_equal(facecolor[0],user_facecolor):
            a = df1['y'][df1['x'] == df3['x'].unique()[sigcount]]
            b = df2['y'][df2['x'] == df3['x'].unique()[sigcount]]

            t_stat, p_value = mannwhitneyu(a, b, alternative='less')
            alpha = 0.05
            if p_value < alpha:
                ax.collections[linecount-1].set_facecolor(to_rgba('lightcoral',alpha=0.5))
            else:
                ax.collections[linecount-1].set_facecolor(to_rgba('darkseagreen',alpha=0.5))
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def printResults(results):
    df = pd.DataFrame(results)
    html_table = df.to_html(index=False, justify="center")
    return html_table
    
if __name__ == '__main__':
    app.run()
