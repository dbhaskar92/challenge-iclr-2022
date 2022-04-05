from geomstats.learning.kmeans import RiemannianKMeans
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.datasets.prepare_graph_data import HyperbolicEmbedding


def PoincareKMeans(embeddings, n_clusters, n_dim=2):
    hyperbolic_manifold = PoincareBall(n_dim)
    kmeans = RiemannianKMeans(
        metric=hyperbolic_manifold.metric,
        n_clusters=n_clusters,
        init="random",
        mean_method="batch",
    )
    centroids = kmeans.fit(X=embeddings)
    labels = kmeans.predict(X=embeddings)
    return (centroids,labels)

def graph_hyperbolic_embedding(graph, num_epochs=20):

    hyperbolic_embedding = HyperbolicEmbedding(max_epochs=num_epochs)
    return hyperbolic_embedding.embed(graph)
    