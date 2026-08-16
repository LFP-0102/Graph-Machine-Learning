"""节点嵌入降维可视化子包。"""

from vis_tool.embedding.tsne import plot_tsne
from vis_tool.embedding.umap import plot_umap


def plot_embeddings(embeddings, labels, method="tsne", **kwargs):
    """按 ``method`` 调用 t-SNE 或 UMAP，保留旧接口以兼容已有代码。"""
    if method.lower() == "umap":
        return plot_umap(embeddings, labels, **kwargs)
    if method.lower() == "tsne":
        return plot_tsne(embeddings, labels, **kwargs)
    raise ValueError(f"未知的降维方法：{method}。可选值：'tsne'、'umap'")


__all__ = ["plot_embeddings", "plot_tsne", "plot_umap"]
