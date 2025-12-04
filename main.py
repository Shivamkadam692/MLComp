from src import preprocess, supervised, unsupervised

results_storage = {}

def run():
    print("=== Comparative ML Analysis ===\n")

    try:
        # Structured Dataset 1: Iris
        print("[+] Loading Iris Dataset...")
        iris_df = preprocess.load_iris()
        print(f"Iris dataset shape: {iris_df.shape}")
        print(f"Iris columns: {list(iris_df.columns)}")
        iris_results = supervised.train_supervised_models(iris_df, 'target')
        results_storage['iris'] = iris_results

        # Structured Dataset 2: Titanic
        print("[+] Loading Titanic Dataset...")
        titanic_df = preprocess.load_titanic()
        print(f"Titanic dataset shape: {titanic_df.shape}")
        print(f"Titanic columns: {list(titanic_df.columns)}")
        titanic_results = supervised.train_supervised_models(titanic_df, 'Survived')
        results_storage['titanic'] = titanic_results

        # Unstructured Dataset: IMDb Sentiment
        print("[+] Loading IMDb Dataset...")
        imdb_df = preprocess.load_imdb()
        print(f"IMDb dataset shape: {imdb_df.shape}")
        print(f"IMDb columns: {list(imdb_df.columns)}")
        imdb_results = supervised.train_sentiment_analysis(imdb_df)
        results_storage['imdb'] = imdb_results

        # Unsupervised Clustering: K-Means on Iris
        print("[+] Running K-Means Clustering...")
        kmeans_results = unsupervised.kmeans_clustering(iris_df.drop('target', axis=1))
        results_storage['kmeans'] = kmeans_results
        
        return results_storage
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        results = run()
        print("\n=== Analysis Complete ===")
    except Exception as e:
        print(f"Application failed: {str(e)}")
