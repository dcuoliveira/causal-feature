import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def simple_lr_plot(df,target_column, feature_column, figsize=(14,5)):
    formula = "{} ~ {}".format(target_column, feature_column)
    lr = smf.ols(formula=formula, data=df).fit()
    r_2 = lr.rsquared
    resid = lr.resid
    resid.name = "residuals"
    resid.index.name = "rindex" 
    resid = resid.reset_index()
    beta0, beta1 = lr.params

    title = "{} ~ {}\n($R^2$ = {:.5f} | $beta_1$ = {:.5f})".format(
            target_column, feature_column, r_2,  beta1)

    X, y = df[feature_column].values, df[target_column].values

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].scatter(X, y,  color="turquoise", alpha=0.5, label="data")
    ax[0].plot(X,beta0 + (X*beta1), color='k',
             linestyle="--",
             linewidth=1.8, label="regression line")
    ax[0].set_title(title, fontsize=18);
    ax[0].set_xlabel(feature_column, fontsize=16);
    ax[0].set_ylabel(target_column, fontsize=16);
    ax[0].legend(loc="upper left");


    first, last = resid.rindex.values[0],resid.rindex.values[-1]

    ax1 = resid.plot.scatter(x='rindex',
                             y='residuals',
                             c="turquoise",
                             alpha=0.5,
                             ax=ax[1])
    ax[1].hlines(0,first, last, color="k", linestyle="--", linewidth=1.8)
    ax[1].set_xlabel('data index', fontsize=16);
    ax[1].set_ylabel('residuals', fontsize=16);
    ax[1].set_title("Residual Plot", fontsize=18);
    plt.subplots_adjust(hspace=0.2, wspace=0.3);
