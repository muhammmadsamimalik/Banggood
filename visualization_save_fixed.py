import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from analysis_basic import load_cleaned_data

# Create visualizations directory
os.makedirs("visualizations", exist_ok=True)

# Set style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_palette("husl")

class EcommerceVisualizer:
    def __init__(self, df):
        self.df = df
    
    def save_plot(self, filename):
        """Helper function to save and close plot"""
        plt.savefig(f'visualizations/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filename}")
    
    def create_price_distribution(self):
        """Price distribution across categories"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        sns.boxplot(data=self.df, x='category_source', y='price', ax=axes[0])
        axes[0].set_title('Price Distribution by Category')
        axes[0].set_xlabel('Category')
        axes[0].set_ylabel('Price ($)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Violin plot
        sns.violinplot(data=self.df, x='category_source', y='price', ax=axes[1])
        axes[1].set_title('Price Density by Category')
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('Price ($)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.save_plot('price_distribution.png')
    
    def create_rating_analysis(self):
        """Rating vs Price scatter plot"""
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(self.df['price'], self.df['rating'], 
                            c=self.df['review_count'], 
                            cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Review Count')
        plt.xlabel('Price ($)')
        plt.ylabel('Rating')
        plt.title('Rating vs Price (Size = Review Count)')
        
        # Add trend line
        z = np.polyfit(self.df['price'], self.df['rating'], 1)
        p = np.poly1d(z)
        plt.plot(self.df['price'], p(self.df['price']), "r--", alpha=0.8)
        
        plt.grid(True, alpha=0.3)
        self.save_plot('rating_vs_price.png')
    
    def create_discount_analysis(self):
        """Discount impact analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Only products with discounts
        discounted_df = self.df[self.df['discount'].notna()]
        
        if not discounted_df.empty:
            # Discount distribution
            axes[0].hist(discounted_df['discount'], bins=20, alpha=0.7, color='skyblue')
            axes[0].set_xlabel('Discount Percentage')
            axes[0].set_ylabel('Number of Products')
            axes[0].set_title('Discount Distribution')
            axes[0].grid(True, alpha=0.3)
            
            # Discount vs Rating
            axes[1].scatter(discounted_df['discount'], discounted_df['rating'], alpha=0.6)
            axes[1].set_xlabel('Discount (%)')
            axes[1].set_ylabel('Rating')
            axes[1].set_title('Discount Impact on Ratings')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_plot('discount_analysis.png')
    
    def create_category_comparison(self):
        """Multi-metric category comparison"""
        categories = self.df['category_source'].unique()
        
        # Calculate metrics
        metrics_data = []
        for category in categories:
            cat_data = self.df[self.df['category_source'] == category]
            metrics_data.append({
                'Category': category,
                'Avg Price': cat_data['price'].mean(),
                'Avg Rating': cat_data['rating'].mean(),
                'Total Reviews': cat_data['review_count'].sum(),
                'Product Count': len(cat_data)
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Avg Price
        axes[0,0].bar(metrics_df['Category'], metrics_df['Avg Price'], color='lightcoral')
        axes[0,0].set_title('Average Price by Category')
        axes[0,0].set_ylabel('Price ($)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Avg Rating
        axes[0,1].bar(metrics_df['Category'], metrics_df['Avg Rating'], color='lightgreen')
        axes[0,1].set_title('Average Rating by Category')
        axes[0,1].set_ylabel('Rating')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Total Reviews
        axes[1,0].bar(metrics_df['Category'], metrics_df['Total Reviews'], color='lightblue')
        axes[1,0].set_title('Total Reviews by Category')
        axes[1,0].set_ylabel('Review Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Product Count
        axes[1,1].bar(metrics_df['Category'], metrics_df['Product Count'], color='gold')
        axes[1,1].set_title('Product Count by Category')
        axes[1,1].set_ylabel('Number of Products')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.save_plot('category_comparison.png')
    
    def create_top_products(self):
        """Show top rated and best value products"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top rated products (min 10 reviews)
        rated_df = self.df[self.df['review_count'] >= 10]
        if not rated_df.empty:
            top_rated = rated_df.nlargest(8, 'rating')[['product_name', 'rating', 'price', 'category_source']]
            
            # Plot top rated
            bars1 = axes[0].barh(range(len(top_rated)), top_rated['rating'], color='lightgreen')
            axes[0].set_yticks(range(len(top_rated)))
            axes[0].set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in top_rated['product_name']])
            axes[0].set_xlabel('Rating')
            axes[0].set_title('Top Rated Products (min 10 reviews)')
            
            # Add price annotations
            for i, (bar, price) in enumerate(zip(bars1, top_rated['price'])):
                axes[0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                            f'${price:.0f}', ha='left', va='center')
        
        # Top value products
        if 'value_score' in self.df.columns:
            top_value = self.df.nlargest(8, 'value_score')[['product_name', 'value_score', 'price', 'category_source']]
            
            # Plot best value
            bars2 = axes[1].barh(range(len(top_value)), top_value['value_score'], color='lightblue')
            axes[1].set_yticks(range(len(top_value)))
            axes[1].set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in top_value['product_name']])
            axes[1].set_xlabel('Value Score')
            axes[1].set_title('Best Value Products')
            
            # Add price annotations
            for i, (bar, price) in enumerate(zip(bars2, top_value['price'])):
                axes[1].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                            f'${price:.0f}', ha='left', va='center')
        
        plt.tight_layout()
        self.save_plot('top_products.png')
    
    def create_brand_analysis(self):
        """Brand performance analysis"""
        # Extract brands (simple method)
        def extract_brand(name):
            if pd.isna(name):
                return 'Unknown'
            name_lower = str(name).lower()
            brands = ['samsung', 'apple', 'xiaomi', 'huawei', 'lenovo', 'dell', 'hp', 'asus']
            for brand in brands:
                if brand in name_lower:
                    return brand.title()
            return 'Other'
        
        self.df['brand'] = self.df['product_name'].apply(extract_brand)
        
        # Brand statistics
        brand_stats = self.df.groupby('brand').agg({
            'price': 'mean',
            'rating': 'mean',
            'review_count': 'sum',
            'product_name': 'count'
        }).rename(columns={'product_name': 'count'})
        
        # Plot brand comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Brand count (top 10)
        brand_stats.nlargest(10, 'count')['count'].plot(kind='bar', ax=axes[0,0], color='lightcoral')
        axes[0,0].set_title('Products per Brand (Top 10)')
        axes[0,0].set_ylabel('Number of Products')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Average price (top 10)
        brand_stats.nlargest(10, 'price')['price'].plot(kind='bar', ax=axes[0,1], color='lightblue')
        axes[0,1].set_title('Average Price by Brand (Top 10)')
        axes[0,1].set_ylabel('Price ($)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Average rating (top 10)
        brand_stats.nlargest(10, 'rating')['rating'].plot(kind='bar', ax=axes[1,0], color='lightgreen')
        axes[1,0].set_title('Average Rating by Brand (Top 10)')
        axes[1,0].set_ylabel('Rating')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Total reviews (top 10)
        brand_stats.nlargest(10, 'review_count')['review_count'].plot(kind='bar', ax=axes[1,1], color='gold')
        axes[1,1].set_title('Total Reviews by Brand (Top 10)')
        axes[1,1].set_ylabel('Review Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.save_plot('brand_analysis.png')
    
    def create_all_visualizations(self):
        """Generate all visualizations"""
        print("📊 Creating and saving visualizations...")
        
        self.create_price_distribution()
        self.create_rating_analysis()
        self.create_discount_analysis()
        self.create_category_comparison()
        self.create_top_products()
        self.create_brand_analysis()
        
        print("\n🎉 All visualizations saved to /visualizations/ folder")
        print("📁 Check the 'visualizations' folder for PNG images")

def main():
    print("🚀 STARTING VISUALIZATION PIPELINE...")
    
    # Load data
    df = load_cleaned_data()
    if df is None or df.empty:
        print("❌ No cleaned data found! Run data_cleaning.py first.")
        return
    
    print(f"📊 Loaded {len(df)} products for visualization")
    
    # Create visualizations
    viz = EcommerceVisualizer(df)
    viz.create_all_visualizations()

if __name__ == "__main__":
    main()