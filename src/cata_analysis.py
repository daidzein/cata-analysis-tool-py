import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.stats.contingency_tables import cochrans_q
import prince
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import warnings

# warnings suppression for cleaner output
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Load and preprocess the data efficiently."""
    # optimizeed dtypes for loading
    # Consumer column named as 'Consumer' and Product column as 'Product'
    dtype_dict = {'Consumer': 'category', 'Product': 'category', 'Liking': 'float32'}
    df = pd.read_csv(filepath, dtype=dtype_dict)
    
    # boolean conversion for memeory efficient cata attributes
    cata_attrs = df.columns[3:]
    df[cata_attrs] = df[cata_attrs].astype('bool')

    # drops rows with missing liking (exclusion of ideal product)
    mask = ~df['Liking'].isna() | (df['Product'] == 'Ideal')
    return df[mask].copy()

def analyze_frequencies(df):
    """Calculate and visualize attribute frequencies."""
    cata_attrs = df.columns[3:]

    # aggregated analysis grouped by product
    product_cata = df.groupby('Product')[cata_attrs].mean()
    product_liking = df.groupby('Product')['Liking'].mean()

    # freq analysis
    frequency_by_product = product_cata * 100
    overall_frequency = product_cata.mean().sort_values(ascending=False) * 100

    # visualization for better comprehension
    fig, axes = plt.subplots(1, 2, figsize=(18, 6)) 
    overall_frequency.plot(kind='bar', ax=axes[0])
    frequency_by_product.plot(kind='bar', ax=axes[1])
    axes[0].set_title('Overall CATA Attribute Frequencies')
    axes[0].set_ylabel('Percentage of Checks')

    sns.heatmap(product_cata.T, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title('CATA Attribute Frequencies by Product')
    axes[1].set_ylabel('CATA Attribute')

    plt.tight_layout()
    plt.show()

    return product_cata, overall_frequency

def cochrans_q_test_for_attribute(attribute, df, products_to_compare):
    """helper function for parallel Cochran's Q tests."""
    try:
        binary_matrix = df.pivot(index='Consumer',
                                 columns='Product',
                                 values=attribute)[products_to_compare].dropna()
        table = binary_matrix.astype(int).values
        result = cochrans_q(table)
        return {
            'Q_statistic': None,
            'p_value': result.pvalue,
            'significant': result.pvalue < 0.05,
            'n_consumers': len(binary_matrix)
        }
    except Exception as e:
        print(f"Error with {attribute}: {str(e)}")
        return {
            'Q_statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e)
        }

def run_cochrans_q_tests(df, max workers=4):
      """We will run parallel Cochran's Q tests for efficiency."""
      products_to_compare = [p for p in df['Product'].unique() if p != 'Ideal']
      cata_attrs = df.columns[3:]

      # utilize parallel processing for the tests
      with ProcessPoolExecutor(max_workers=max_workers) as executor:
        test_func = partial(cochrans_q_test_for_attribute, df=df,
                          products_to_compare=products_to_compare)
          results = list(executor.map(test_func, cata_attrs))

      return pd.DataFrame(results, index=cata_attrs)

def penalty_analysis(df):
    """Efficiently perform penalty analysis."""
    cata_attrs = df.columns[3:]
    analysis_df = df[['Product', 'Consumer', 'Liking'] + list(cata_attrs)]

    # Melt the dataframe for efficient grouping
    # if your attribute data presented as long format, you wouldn't need to melt the dataframe
    melted = analysis_df.melt(id_vars=['Product', 'Consumer', 'Liking'],
                            var_name='Attribute', value_name='Present')

    # Calculate mean liking by attribute presence
    grouped = melted.groupby(['Attribute', 'Present'])['Liking'].agg(['mean', 'size'])
    grouped = grouped.unstack()

    # Calculate penalty scores
    penalty_df = pd.DataFrame({
        'Attribute': grouped.index,
        'Mean Liking when Present': grouped[('mean', True)],
        'Mean Liking when Absent': grouped[('mean', False)],
        'Penalty': grouped[('mean', False)] - grouped[('mean', True)],
        'Occurrences': grouped[('size', True)]
    }).sort_values('Penalty', ascending=False)

    # Visualization using Seaborn
    plt = sns.mpl.pyplot
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=penalty_df, x='Occurrences', y='Penalty')
    for i, row in penalty_df.iterrows():
        plt.text(row['Occurrences'], row['Penalty'], row['Attribute'],
                ha='center', va='bottom')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Penalty Analysis')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Penalty (Absent - Present)')
    plt.tight_layout()
    plt.show()

    return penalty_df

def correspondence_analysis(product_cata):
    """Perform correspondence analysis and visualization."""
    ca = prince.CA(n_components=2)
    ca = ca.fit(product_cata)

    row_coords = ca.row_coordinates(product_cata)
    col_coords = ca.column_coordinates(product_cata)

    plt = sns.mpl.pyplot
    plt.figure(figsize=(12, 8))
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)

    sns.scatterplot(x=row_coords[0], y=row_coords[1], color='red', label='Products')
    sns.scatterplot(x=col_coords[0], y=col_coords[1], color='blue', label='Attributes')

    for i, txt in enumerate(row_coords.index):
        plt.annotate(txt, (row_coords.iloc[i, 0], row_coords.iloc[i, 1]))
    for i, txt in enumerate(col_coords.index):
        plt.annotate(txt, (col_coords.iloc[i, 0], col_coords.iloc[i, 1]))

    plt.title("Correspondence Analysis of Products and Attributes")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return ca

def main():
    # Load and preprocess data
    # insert the name of your CSV data into ('insert_data_here')
    df = load_and_preprocess_data('insert_data_here.csv')

    # Basic frequency analysis
    product_cata, overall_freq = analyze_frequencies(df)
    print("\nOverall attribute frequency (%):")
    print(overall_freq)

    # Cochran's Q tests
    q_results = run_cochrans_q_tests(df)
    print("\nCochran's Q Test Results:")
    print(q_results)

    # Penalty analysis
    penalty_df = penalty_analysis(df)
    print("\nPenalty Analysis Results:")
    print(penalty_df)

    # Correspondence analysis
    ca = correspondence_analysis(product_cata)

    # Difference from Ideal analysis
    if 'Ideal' in product_cata.index:
        ideal_cata = product_cata.loc['Ideal']
        ideal_comparison = product_cata.sub(ideal_cata, axis=1)

        plt = sns.mpl.pyplot
        plt.figure(figsize=(12, 8))
        sns.heatmap(ideal_comparison.T,
                   annot=True,
                   fmt=".2f",
                   cmap="coolwarm",
                   center=0,
                   vmin=-1,
                   vmax=1)
        plt.title('Difference from Ideal Product')
        plt.tight_layout()
        plt.show()

    # Perform PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(product_cata)

    # Create biplot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=product_cata.index)

    # Add attribute vectors
    for i, feature in enumerate(product_cata.columns):
        plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
                  color='r', alpha=0.5)
        plt.text(pca.components_[0, i]*1.1, pca.components_[1, i]*1.1,
                 feature, color='r')

    plt.title('PCA Biplot of Coffee Products and Attributes')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()