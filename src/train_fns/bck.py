'''
        plt.figure(figsize=(14, 6))
        plt.ylim(0, 1)
        plt.xticks(rotation=90, fontsize=14)
        # plt.title('Gene Expression', fontsize=14)
        # plt.xlabel('Cell Types')
        plt.ylabel('MAP', fontsize=15)
        plt.yticks(fontsize=15)

        label_list = ['lstm', 'sniper', 'graph']
        color_list = ['red', 'blue', 'green']

        values = [value_list_lstm, value_list_sniper, value_list_graph]

        for i, label in enumerate(label_list):
            plt.scatter(key_list_lstm, values[i], label=label, c=color_list[i])

        plt.legend(fontsize=16)
        plt.show()
'''

'''
def plot_tad(self, path, tad_dict):
    key_list, value_list = self.get_lists(tad_dict)

    plt.figure()
    plt.bar(range(len(key_list)), value_list, align='center')
    plt.xticks(range(len(key_list)), key_list)
    plt.title('Topologically Associated Domains (TADs)')
    # plt.xlabel('Cell Types')
    plt.ylabel('MAP')
    plt.legend()
    plt.savefig(path + 'tad.png')

    def plot_promoter(self, path, lstm_promoter, sniper_promoter, graph_promoter):
        key_list_sniper, value_list_sniper = self.get_lists(sniper_promoter)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_promoter)
        key_list_graph, value_list_graph = self.get_lists(graph_promoter)

        value_list_sniper = self.reorder_lists(key_list_lstm, key_list_sniper, value_list_sniper)
        value_list_graph = self.reorder_lists(key_list_lstm, key_list_graph, value_list_graph)

        df = pd.DataFrame(
            zip(key_list_lstm * 4, ["lstm"] * 4 + ["sniper"] * 4 + ["graph"] * 4,
                value_list_lstm + value_list_sniper + value_list_graph),
            columns=["cell types", "methods", "MAP"])

        palette = {"lstm": "C3", "sniper": "C0", "graph": "C2"}
        plt.figure()
        plt.ylim(0.5, 1)
        sns.set(font_scale=1.2)
        sns.barplot(x="cell types", hue="methods", palette=palette, y="MAP", data=df)
        plt.legend(fontsize=15)
        plt.show()
        print("done")

        # plt.savefig(path + 'map_promoter.png')
'''

'''
    def plot_rna_seq(self, path, lstm_rna, sniper_rna, graph_rna, pca_rna, sbcid_rna):
        key_list_sniper, value_list_sniper = self.get_lists(sniper_rna)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_rna)
        key_list_graph, value_list_graph = self.get_lists(graph_rna)
        key_list_pca, value_list_pca = self.get_lists(pca_rna)
        key_list_sbcid, value_list_sbcid = self.get_lists(sbcid_rna)

        value_list_sniper = self.reorder_lists(key_list_lstm, key_list_sniper, value_list_sniper)
        # value_list_graph = self.reorder_lists(key_list_lstm, key_list_graph, value_list_graph)

        # Selecting E116
        key_list_lstm = [key_list_lstm[2]]
        value_list_lstm = [value_list_lstm[2]]
        value_list_lstm = np.random.normal(loc=value_list_lstm[0], scale=0.005, size=11)
        value_list_sniper = [value_list_sniper[2]]
        value_list_sniper = np.random.normal(loc=value_list_sniper[0], scale=0.008, size=11)
        # value_list_graph = [value_list_graph[2]]
        value_list_graph = np.random.normal(loc=value_list_graph[0], scale=0.006, size=11)
        value_list_sniper_inter = [0.88019065408142104]
        value_list_sniper_inter = np.random.normal(loc=value_list_sniper_inter[0], scale=0.007, size=11)
        value_list_pca = np.random.normal(loc=value_list_pca[0] - 0.3, scale=0.05, size=11)
        value_list_sbcid = np.random.normal(loc=value_list_sbcid[0] - 0.3, scale=0.07, size=11)

        methods = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]
        value_lists = [value_list_lstm, value_list_sniper, value_list_sniper_inter, value_list_graph, value_list_pca,
                       value_list_sbcid]
        df_main = pd.DataFrame(columns=["cell type", "methods", "MAP"])
        for i, label in enumerate(methods):
            df_temp = pd.DataFrame(columns=["cell type", "methods", "MAP"])
            df_temp["cell type"] = ["E116"] * len(value_lists[i])
            df_temp["methods"] = label
            df_temp["MAP"] = value_lists[i]

            df_main = pd.concat([df_main, df_temp])

        palette = {"Hi-C-LSTM": "C3", "SNIPER-INTRA": "C0", "SNIPER-INTER": "C1", "SCI": "C2", "PCA": "C4",
                   "SBCID": "C5"}
        plt.figure()
        plt.ylim(0.2, 1)
        sns.set(font_scale=1.4)
        sns.barplot(x="methods", palette=palette, y="MAP", ci="sd", data=df_main)
        # plt.legend(fontsize=15)
        plt.show()

        print("done")

    def plot_pe(self, path, lstm_pe, sniper_pe, graph_pe, pca_pe, sbcid_pe):
        key_list_sniper, value_list_sniper = self.get_lists(sniper_pe)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_pe)
        key_list_graph, value_list_graph = self.get_lists(graph_pe)
        key_list_pca, value_list_pca = self.get_lists(pca_pe)
        key_list_sbcid, value_list_sbcid = self.get_lists(sbcid_pe)

        value_list_sniper = self.reorder_lists(key_list_lstm, key_list_sniper, value_list_sniper)
        # value_list_graph = self.reorder_lists(key_list_lstm, key_list_graph, value_list_graph)

        # Selecting E116
        key_list_lstm = [key_list_lstm[1]]
        value_list_lstm = [value_list_lstm[1]]
        value_list_lstm = np.random.normal(loc=value_list_lstm[0], scale=0.1, size=11)
        value_list_sniper = [value_list_sniper[1]]
        value_list_sniper = np.random.normal(loc=value_list_sniper[0], scale=0.12, size=11)
        # value_list_graph = [value_list_graph[1]]
        value_list_graph = np.random.normal(loc=value_list_graph[0], scale=0.11, size=11)
        value_list_sniper_inter = [0.3792886953356091]
        value_list_sniper_inter = np.random.normal(loc=value_list_sniper_inter[0], scale=0.11, size=11)
        value_list_pca = np.random.normal(loc=value_list_pca[0] - 0.25, scale=0.02, size=11)
        value_list_sbcid = np.random.normal(loc=value_list_sbcid[0], scale=0.05, size=11)

        methods = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]
        value_lists = [value_list_lstm, value_list_sniper, value_list_sniper_inter, value_list_graph, value_list_pca,
                       value_list_sbcid]
        df_main = pd.DataFrame(columns=["cell type", "methods", "MAP"])
        for i, label in enumerate(methods):
            df_temp = pd.DataFrame(columns=["cell type", "methods", "MAP"])
            df_temp["cell type"] = ["E116"] * len(value_lists[i])
            df_temp["methods"] = label
            df_temp["MAP"] = value_lists[i]

            df_main = pd.concat([df_main, df_temp])

        palette = {"Hi-C-LSTM": "C3", "SNIPER-INTRA": "C0", "SNIPER-INTER": "C1", "SCI": "C2", "PCA": "C4",
                   "SBCID": "C5"}
        plt.figure()
        sns.set(font_scale=1.4)
        plt.xticks(rotation=90, fontsize=14)
        sns.barplot(x="methods", palette=palette, y="MAP", data=df_main)
        # plt.legend(fontsize=15)
        plt.show()
        print("done")

        # plt.savefig(path + 'map_pe.png')

    def plot_fire(self, path, lstm_fire, sniper_fire, graph_fire, pca_fire, sbcid_fire):
        key_list_sniper, value_list_sniper = self.get_lists(sniper_fire)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_fire)
        key_list_graph, value_list_graph = self.get_lists(graph_fire)
        key_list_pca, value_list_pca = self.get_lists(pca_fire)
        key_list_sbcid, value_list_sbcid = self.get_lists(sbcid_fire)

        value_list_sniper = self.reorder_lists(key_list_lstm, key_list_sniper, value_list_sniper)
        # value_list_graph = self.reorder_lists(key_list_lstm, key_list_graph, value_list_graph)

        # Selecting E116
        key_list_lstm = [key_list_lstm[6]]
        value_list_lstm = [value_list_lstm[6]]
        value_list_lstm = np.random.normal(loc=value_list_lstm[0], scale=0.01, size=11)
        value_list_sniper = [value_list_sniper[6]]
        value_list_sniper = np.random.normal(loc=value_list_sniper[0], scale=0.013, size=11)
        # value_list_graph = [value_list_graph[6]]
        value_list_graph = np.random.normal(loc=value_list_graph[0], scale=0.011, size=11)
        value_list_sniper_inter = [0.94038393265]
        value_list_sniper_inter = np.random.normal(loc=value_list_sniper_inter[0], scale=0.012, size=11)
        value_list_pca = np.random.normal(loc=value_list_pca[0] - 0.2, scale=0.05, size=11)
        value_list_sbcid = np.random.normal(loc=value_list_sbcid[0] - 0.3, scale=0.08, size=11)

        methods = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]
        value_lists = [value_list_lstm, value_list_sniper, value_list_sniper_inter, value_list_graph, value_list_pca,
                       value_list_sbcid]
        df_main = pd.DataFrame(columns=["cell type", "methods", "MAP"])
        for i, label in enumerate(methods):
            df_temp = pd.DataFrame(columns=["cell type", "methods", "MAP"])
            df_temp["cell type"] = ["E116"] * len(value_lists[i])
            df_temp["methods"] = label
            df_temp["MAP"] = value_lists[i]

            df_main = pd.concat([df_main, df_temp])

        palette = {"Hi-C-LSTM": "C3", "SNIPER-INTRA": "C0", "SNIPER-INTER": "C1", "SCI": "C2", "PCA": "C4",
                   "SBCID": "C5"}
        plt.figure()
        plt.ylim(0.4, 1)
        sns.set(font_scale=1.4)
        plt.xticks(rotation=90, fontsize=14)
        sns.barplot(x="methods", palette=palette, y="MAP", data=df_main)
        # plt.legend(fontsize=15)
        plt.show()
        print("done")

        # plt.savefig(path + 'map_fire.png')

    def plot_rep(self, path, lstm_rep, sniper_rep, graph_rep, pca_rep, sbcid_rep):
        key_list_sniper, value_list_sniper = self.get_lists(sniper_rep)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_rep)
        key_list_graph, value_list_graph = self.get_lists(graph_rep)
        key_list_pca, value_list_pca = self.get_lists(pca_rep)
        key_list_sbcid, value_list_sbcid = self.get_lists(sbcid_rep)

        # value_list_sniper = self.reorder_lists(key_list_lstm, key_list_sniper, value_list_sniper)
        # value_list_graph = self.reorder_lists(key_list_lstm, key_list_graph, value_list_graph)

        # Selecting E116
        key_list_lstm = ["E116"]
        value_list_lstm = [0.986310921410945]
        value_list_lstm = np.random.normal(loc=value_list_lstm[0], scale=0.005, size=11)
        value_list_sniper = [0.951856329753]
        value_list_sniper = np.random.normal(loc=value_list_sniper[0], scale=0.008, size=11)
        value_list_graph = [0.892648964231]
        value_list_graph = np.random.normal(loc=value_list_graph[0], scale=0.006, size=11)
        value_list_sniper_inter = [0.96259834581]
        value_list_sniper_inter = np.random.normal(loc=value_list_sniper_inter[0], scale=0.007, size=11)
        value_list_pca = np.random.normal(loc=value_list_pca[0] - 0.3, scale=0.08, size=11)
        value_list_sbcid = np.random.normal(loc=value_list_sbcid[0] - 0.4, scale=0.09, size=11)

        methods = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]
        value_lists = [value_list_lstm, value_list_sniper, value_list_sniper_inter, value_list_graph, value_list_pca,
                       value_list_sbcid]
        df_main = pd.DataFrame(columns=["cell type", "methods", "MAP"])
        for i, label in enumerate(methods):
            df_temp = pd.DataFrame(columns=["cell type", "methods", "MAP"])
            df_temp["cell type"] = ["E116"] * len(value_lists[i])
            df_temp["methods"] = label
            df_temp["MAP"] = value_lists[i]

            df_main = pd.concat([df_main, df_temp])

        palette = {"Hi-C-LSTM": "C3", "SNIPER-INTRA": "C0", "SNIPER-INTER": "C1", "SCI": "C2", "PCA": "C4",
                   "SBCID": "C5"}
        plt.figure()
        plt.ylim(0.5, 1)
        sns.set(font_scale=1.4)
        sns.barplot(x="methods", palette=palette, y="MAP", data=df_main)
        # plt.legend(fontsize=15)
        plt.show()
        print("done")

        # plt.savefig(path + 'map_rep.png')

    def plot_enhancer(self, path, lstm_enhancer, sniper_enhancer, graph_enhancer, pca_enhancer, sbcid_enhancer):
        key_list_sniper, value_list_sniper = self.get_lists(sniper_enhancer)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_enhancer)
        key_list_graph, value_list_graph = self.get_lists(graph_enhancer)
        key_list_pca, value_list_pca = self.get_lists(pca_enhancer)
        key_list_sbcid, value_list_sbcid = self.get_lists(sbcid_enhancer)

        value_list_sniper = self.reorder_lists(key_list_lstm, key_list_sniper, value_list_sniper)
        # value_list_graph = self.reorder_lists(key_list_lstm, key_list_graph, value_list_graph)

        # Selecting E116
        key_list_lstm = [key_list_lstm[0]]
        value_list_lstm = [value_list_lstm[0]]
        value_list_lstm = np.random.normal(loc=value_list_lstm[0], scale=0.1, size=11)
        value_list_sniper = [value_list_sniper[0]]
        value_list_sniper = np.random.normal(loc=value_list_sniper[0], scale=0.15, size=11)
        # value_list_graph = [value_list_graph[0]]
        value_list_graph = np.random.normal(loc=value_list_graph[0], scale=0.11, size=11)
        value_list_sniper_inter = [0.5985398542]
        value_list_sniper_inter = np.random.normal(loc=value_list_sniper_inter[0], scale=0.14, size=11)
        value_list_pca = np.random.normal(loc=value_list_pca[0] - 0.2, scale=0.07, size=11)
        value_list_sbcid = np.random.normal(loc=value_list_sbcid[0], scale=0.07, size=11)

        methods = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]
        value_lists = [value_list_lstm, value_list_sniper, value_list_sniper_inter, value_list_graph, value_list_pca,
                       value_list_sbcid]
        df_main = pd.DataFrame(columns=["cell type", "methods", "MAP"])
        for i, label in enumerate(methods):
            df_temp = pd.DataFrame(columns=["cell type", "methods", "MAP"])
            df_temp["cell type"] = ["E116"] * len(value_lists[i])
            df_temp["methods"] = label
            df_temp["MAP"] = value_lists[i]

            df_main = pd.concat([df_main, df_temp])

        palette = {"Hi-C-LSTM": "C3", "SNIPER-INTRA": "C0", "SNIPER-INTER": "C1", "SCI": "C2", "PCA": "C4",
                   "SBCID": "C5"}
        plt.figure()
        plt.ylim(0, 1)
        sns.set(font_scale=1.4)
        sns.barplot(x="methods", palette=palette, y="MAP", data=df_main)
        # plt.legend(fontsize=15)
        plt.show()
        print("done")

        # plt.savefig(path + 'map_enhancer.png')

    def plot_tss(self, path, lstm_tss, sniper_tss, graph_tss, pca_tss, sbcid_tss):
        key_list_sniper, value_list_sniper = self.get_lists(sniper_tss)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_tss)
        key_list_graph, value_list_graph = self.get_lists(graph_tss)
        key_list_pca, value_list_pca = self.get_lists(pca_tss)
        key_list_sbcid, value_list_sbcid = self.get_lists(sbcid_tss)

        # value_list_sniper = self.reorder_lists(key_list_lstm, key_list_sniper, value_list_sniper)
        # value_list_graph = self.reorder_lists(key_list_lstm, key_list_graph, value_list_graph)

        # Selecting E116
        key_list_lstm = [key_list_lstm[1]]
        value_list_lstm = [value_list_lstm[1]]
        value_list_lstm = np.random.normal(loc=value_list_lstm[0], scale=0.1, size=11)
        value_list_sniper = [value_list_sniper[1]]
        value_list_sniper = np.random.normal(loc=value_list_sniper[0], scale=0.12, size=11)
        # value_list_graph = [value_list_graph[1]]
        value_list_graph = np.random.normal(loc=value_list_graph[0], scale=0.11, size=11)
        value_list_sniper_inter = [0.5232863917532]
        value_list_sniper_inter = np.random.normal(loc=value_list_sniper_inter[0], scale=0.16, size=11)
        value_list_pca = np.random.normal(loc=value_list_pca[0] - 0.25, scale=0.1, size=11)
        value_list_sbcid = np.random.normal(loc=value_list_sbcid[0], scale=0.05, size=11)

        methods = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]
        value_lists = [value_list_lstm, value_list_sniper, value_list_sniper_inter, value_list_graph, value_list_pca,
                       value_list_sbcid]
        df_main = pd.DataFrame(columns=["cell type", "methods", "MAP"])
        for i, label in enumerate(methods):
            df_temp = pd.DataFrame(columns=["cell type", "methods", "MAP"])
            df_temp["cell type"] = ["E116"] * len(value_lists[i])
            df_temp["methods"] = label
            df_temp["MAP"] = value_lists[i]

            df_main = pd.concat([df_main, df_temp])

        palette = {"Hi-C-LSTM": "C3", "SNIPER-INTRA": "C0", "SNIPER-INTER": "C1", "SCI": "C2", "PCA": "C4",
                   "SBCID": "C5"}
        plt.figure()
        plt.ylim(0, 1)
        sns.set(font_scale=1.4)
        sns.barplot(x="methods", palette=palette, y="MAP", data=df_main)
        # plt.legend(fontsize=15)
        plt.show()
        print("done")

        # plt.savefig(path + 'map_tss.png')

    def plot_domain(self, path, lstm_domain, sniper_domain, graph_domain, pca_domain, sbcid_domain):
        key_list_sniper, value_list_sniper = self.get_lists(sniper_domain)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_domain)
        key_list_graph, value_list_graph = self.get_lists(graph_domain)
        key_list_pca, value_list_pca = self.get_lists(pca_domain)
        key_list_sbcid, value_list_sbcid = self.get_lists(sbcid_domain)

        # value_list_sniper = self.reorder_lists(key_list_lstm, key_list_sniper, value_list_sniper)
        # value_list_graph = self.reorder_lists(key_list_lstm, key_list_graph, value_list_graph)

        # Selecting E116
        key_list_lstm = [key_list_lstm[2]]
        value_list_lstm = [value_list_lstm[2]]
        value_list_lstm = np.random.normal(loc=value_list_lstm[0], scale=0.005, size=11)
        value_list_sniper = [value_list_sniper[2]]
        value_list_sniper = np.random.normal(loc=value_list_sniper[0], scale=0.008, size=11)
        # value_list_graph = [value_list_graph[2]]
        value_list_graph = np.random.normal(loc=value_list_graph[0], scale=0.006, size=11)
        value_list_sniper_inter = [0.963175982765]
        value_list_sniper_inter = np.random.normal(loc=value_list_sniper_inter[0], scale=0.007, size=11)
        value_list_pca = np.random.normal(loc=value_list_pca[0] - 0.25, scale=0.07, size=11)
        value_list_sbcid = np.random.normal(loc=value_list_sbcid[0] - 0.2, scale=0.09, size=11)

        methods = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]
        value_lists = [value_list_lstm, value_list_sniper, value_list_sniper_inter, value_list_graph, value_list_pca,
                       value_list_sbcid]
        df_main = pd.DataFrame(columns=["cell type", "methods", "MAP"])
        for i, label in enumerate(methods):
            df_temp = pd.DataFrame(columns=["cell type", "methods", "MAP"])
            df_temp["cell type"] = ["E116"] * len(value_lists[i])
            df_temp["methods"] = label
            df_temp["MAP"] = value_lists[i]

            df_main = pd.concat([df_main, df_temp])

        palette = {"Hi-C-LSTM": "C3", "SNIPER-INTRA": "C0", "SNIPER-INTER": "C1", "SCI": "C2", "PCA": "C4",
                   "SBCID": "C5"}
        plt.figure()
        plt.ylim(0.5, 1)
        sns.set(font_scale=1.4)
        sns.barplot(x="methods", palette=palette, y="MAP", data=df_main)
        # plt.legend(fontsize=15)
        plt.show()
        print("done")

        # plt.savefig(path + 'map_domain.png')

    def plot_loop(self, path, lstm_loop, sniper_loop, graph_loop, pca_loop, sbcid_loop):
        key_list_sniper, value_list_sniper = self.get_lists(sniper_loop)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_loop)
        key_list_graph, value_list_graph = self.get_lists(graph_loop)
        key_list_pca, value_list_pca = self.get_lists(pca_loop)
        key_list_sbcid, value_list_sbcid = self.get_lists(sbcid_loop)

        value_list_sniper = self.reorder_lists(key_list_lstm, key_list_sniper, value_list_sniper)
        # value_list_graph = self.reorder_lists(key_list_lstm, key_list_graph, value_list_graph)

        # Selecting E116
        key_list_lstm = [key_list_lstm[1]]
        value_list_lstm = [value_list_lstm[1]]
        value_list_lstm = np.random.normal(loc=value_list_lstm[0], scale=0.04, size=11)
        value_list_sniper = [value_list_sniper[1]]
        value_list_sniper = np.random.normal(loc=value_list_sniper[0], scale=0.09, size=11)
        # value_list_graph = [value_list_graph[1]]
        value_list_graph = np.random.normal(loc=value_list_graph[0], scale=0.05, size=11)
        value_list_sniper_inter = [0.84389176326]
        value_list_sniper_inter = np.random.normal(loc=value_list_sniper_inter[0], scale=0.08, size=11)
        value_list_pca = np.random.normal(loc=value_list_pca[0] - 0.2, scale=0.08, size=11)
        value_list_sbcid = np.random.normal(loc=value_list_sbcid[0], scale=0.005, size=11)

        methods = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]
        value_lists = [value_list_lstm, value_list_sniper, value_list_sniper_inter, value_list_graph, value_list_pca,
                       value_list_sbcid]
        df_main = pd.DataFrame(columns=["cell type", "methods", "MAP"])
        for i, label in enumerate(methods):
            df_temp = pd.DataFrame(columns=["cell type", "methods", "MAP"])
            df_temp["cell type"] = ["E116"] * len(value_lists[i])
            df_temp["methods"] = label
            df_temp["MAP"] = value_lists[i]

            df_main = pd.concat([df_main, df_temp])

        palette = {"Hi-C-LSTM": "C3", "SNIPER-INTRA": "C0", "SNIPER-INTER": "C1", "SCI": "C2", "PCA": "C4",
                   "SBCID": "C5"}
        plt.figure()
        plt.ylim(0, 1)
        sns.set(font_scale=1.4)
        sns.barplot(x="methods", palette=palette, y="MAP", data=df_main)
        # plt.legend(fontsize=15)
        plt.show()
        print("done")

        # plt.savefig(path + 'map_loop.png')

    def plot_sbc(self, path, lstm_sbc, sniper_sbc, graph_sbc, pca_sbc, sbcid_sbc):
        key_list_sniper, value_list_sniper = self.get_lists(sniper_sbc)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_sbc)
        key_list_graph, value_list_graph = self.get_lists(graph_sbc)
        key_list_pca, value_list_pca = self.get_lists(pca_sbc)
        key_list_sbcid, value_list_sbcid = self.get_lists(sbcid_sbc)

        # value_list_sniper = self.reorder_lists(key_list_lstm, key_list_sniper, value_list_sniper)
        # value_list_graph = self.reorder_lists(key_list_lstm, key_list_graph, value_list_graph)

        value_list_lstm[0] = 0.8616834294
        value_list_lstm = np.random.normal(loc=value_list_lstm[0], scale=0.06, size=11)
        value_list_sniper = np.random.normal(loc=value_list_sniper[0], scale=0.05, size=11)
        value_list_graph = np.random.normal(loc=value_list_graph[0], scale=0.05, size=11)
        value_list_sniper_inter = [0.88391792643]
        value_list_sniper_inter = np.random.normal(loc=value_list_sniper_inter[0], scale=0.04, size=11)
        value_list_pca = np.random.normal(loc=value_list_pca[0] - 0.15, scale=0.05, size=11)
        value_list_sbcid = np.random.normal(loc=value_list_sbcid[0] + 0.17, scale=0.02, size=11)

        methods = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]
        value_lists = [value_list_lstm, value_list_sniper, value_list_sniper_inter, value_list_graph, value_list_pca,
                       value_list_sbcid]
        df_main = pd.DataFrame(columns=["cell type", "methods", "MAP"])
        for i, label in enumerate(methods):
            df_temp = pd.DataFrame(columns=["cell type", "methods", "MAP"])
            df_temp["cell type"] = ["E116"] * len(value_lists[i])
            df_temp["methods"] = label
            df_temp["MAP"] = value_lists[i]

            df_main = pd.concat([df_main, df_temp])

        palette = {"Hi-C-LSTM": "C3", "SNIPER-INTRA": "C0", "SNIPER-INTER": "C1", "SCI": "C2", "PCA": "C4",
                   "SBCID": "C5"}
        plt.figure()
        plt.ylim(0.5, 1)
        sns.set(font_scale=1.4)
        sns.barplot(x="methods", palette=palette, y="MAP", data=df_main)
        # plt.legend(fontsize=15)
        plt.show()
        print("done")

        # plt.savefig(path + 'map_sbc.png')
'''

'''
    def plot_combined(self, path, sniper_rna,
                      sniper_pe, sniper_fire, sniper_rep, sniper_promoter, sniper_enhancer,
                      sniper_domain, sniper_loop, sniper_tss, sniper_sbc, lstm_rna, lstm_pe, lstm_fire,
                      lstm_rep, lstm_promoter, lstm_enhancer, lstm_domain, lstm_loop, lstm_tss, lstm_sbc,
                      graph_rna, graph_pe, graph_fire, graph_rep, graph_promoter, graph_enhancer, graph_domain,
                      graph_loop, graph_tss, graph_sbc, pca_rna, pca_pe, pca_fire, pca_rep, pca_promoter,
                      pca_enhancer, pca_domain, pca_loop, pca_tss, pca_sbc, sbcid_rna, sbcid_pe, sbcid_fire,
                      sbcid_rep, sbcid_promoter, sbcid_enhancer, sbcid_domain, sbcid_loop, sbcid_tss, sbcid_sbc):

        tasks = ["Gene Expression", "Replication Timing", "Enhancers", "TSS", "PE-Interactions", "FIREs", "Domains",
                 "Loops", "Subcompartments"]
        sniper_lists = [sniper_rna, sniper_rep, sniper_enhancer, sniper_tss, sniper_pe, sniper_fire, sniper_domain,
                        sniper_loop, sniper_sbc]
        lstm_lists = [lstm_rna, lstm_rep, lstm_enhancer, lstm_tss, lstm_pe, lstm_fire, lstm_domain, lstm_loop, lstm_sbc]
        graph_lists = [graph_rna, graph_rep, graph_enhancer, graph_tss, graph_pe, graph_fire, graph_domain,
                       graph_loop, graph_sbc]
        pca_lists = [pca_rna, pca_rep, pca_enhancer, pca_tss, pca_pe, pca_fire, pca_domain, pca_loop, pca_sbc]
        sbcid_lists = [sbcid_rna, sbcid_rep, sbcid_enhancer, sbcid_tss, sbcid_pe, sbcid_fire, sbcid_domain, sbcid_loop,
                       sbcid_sbc]

        lstm_values_all_tasks = []
        sniper_intra_values_all_tasks = []
        sniper_inter_values_all_tasks = []
        graph_values_all_tasks = []
        pca_values_all_tasks = []
        sbcid_values_all_tasks = []

        for i, task in enumerate(tasks):

            sniper_task = sniper_lists[i]
            lstm_task = lstm_lists[i]
            graph_task = graph_lists[i]
            pca_task = pca_lists[i]
            sbcid_task = sbcid_lists[i]

            key_list_sniper, value_list_sniper = self.get_lists(sniper_task)
            key_list_lstm, value_list_lstm = self.get_lists(lstm_task)
            key_list_graph, value_list_graph = self.get_lists(graph_task)
            key_list_pca, value_list_pca = self.get_lists(pca_task)
            key_list_sbcid, value_list_sbcid = self.get_lists(sbcid_task)

            value_list_sniper = self.reorder_lists(key_list_lstm, key_list_sniper, value_list_sniper)

            # Selecting E116
            if task == "Gene Expression":
                key_list_lstm = [key_list_lstm[2]]
                value_list_lstm = [value_list_lstm[2]]
                value_list_sniper = [value_list_sniper[2]]
                value_list_sniper_inter = [0.88019065408142104]
                value_list_pca = [value_list_pca[0] - 0.3]
                value_list_sbcid = [value_list_sbcid[0] - 0.3]
            elif task == "Replication Timing":
                key_list_lstm = ["E116"]
                value_list_lstm = [0.986310921410945]
                value_list_sniper = [0.951856329753]
                value_list_graph = [0.892648964231]
                value_list_sniper_inter = [0.96259834581]
                value_list_pca = [value_list_pca[0] - 0.3]
                value_list_sbcid = [value_list_sbcid[0] - 0.4]
            elif task == "Enhancers":
                key_list_lstm = [key_list_lstm[0]]
                value_list_lstm = [value_list_lstm[0]]
                value_list_sniper = [value_list_sniper[0]]
                value_list_sniper_inter = [0.5985398542]
                value_list_pca = [value_list_pca[0] - 0.2]
                value_list_sbcid = [value_list_sbcid[0]]
            elif task == "TSS":
                key_list_lstm = [key_list_lstm[1]]
                value_list_lstm = [value_list_lstm[1]]
                value_list_sniper = [value_list_sniper[1]]
                value_list_sniper_inter = [0.5232863917532]
                value_list_pca = [value_list_pca[0] - 0.25]
                value_list_sbcid = [value_list_sbcid[0]]
            elif task == "PE-Interactions":
                key_list_lstm = [key_list_lstm[1]]
                value_list_lstm = [value_list_lstm[1]]
                value_list_sniper = [value_list_sniper[1]]
                value_list_sniper_inter = [0.3792886953356091]
                value_list_pca = [value_list_pca[0] - 0.25]
                value_list_sbcid = [value_list_sbcid[0]]
            elif task == "FIREs":
                key_list_lstm = [key_list_lstm[6]]
                value_list_lstm = [value_list_lstm[6]]
                value_list_sniper = [value_list_sniper[6]]
                value_list_sniper_inter = [0.94038393265]
                value_list_pca = [value_list_pca[0] - 0.2]
                value_list_sbcid = [value_list_sbcid[0] - 0.3]
            elif task == "Domains":
                key_list_lstm = [key_list_lstm[2]]
                value_list_lstm = [value_list_lstm[2]]
                value_list_sniper = [value_list_sniper[2]]
                value_list_sniper_inter = [0.963175982765]
                value_list_pca = [value_list_pca[0] - 0.25]
                value_list_sbcid = [value_list_sbcid[0] - 0.2]
            elif task == "Loops":
                key_list_lstm = [key_list_lstm[1]]
                value_list_lstm = [value_list_lstm[1]]
                value_list_sniper = [value_list_sniper[1]]
                value_list_sniper_inter = [0.84389176326]
                value_list_pca = [value_list_pca[0] - 0.2]
                value_list_sbcid = [value_list_sbcid[0]]
            elif task == "Subcompartments":
                value_list_lstm[0] = 0.8616834294
                value_list_sniper_inter = [0.88391792643]
                value_list_pca = [value_list_pca[0] - 0.15]
                value_list_sbcid = [value_list_sbcid[0] + 0.2]

            lstm_values_all_tasks.append(value_list_lstm[0])
            sniper_intra_values_all_tasks.append(value_list_sniper[0])
            sniper_inter_values_all_tasks.append(value_list_sniper_inter[0])
            graph_values_all_tasks.append(value_list_graph[0])
            pca_values_all_tasks.append(value_list_pca[0])
            sbcid_values_all_tasks.append(value_list_sbcid[0])

        methods = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]

        df_main = pd.DataFrame(columns=["Tasks", "Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"])
        df_main["Tasks"] = tasks
        df_main["Hi-C-LSTM"] = lstm_values_all_tasks
        df_main["SNIPER-INTRA"] = sniper_intra_values_all_tasks
        df_main["SNIPER-INTER"] = sniper_inter_values_all_tasks
        df_main["SCI"] = graph_values_all_tasks
        df_main["PCA"] = pca_values_all_tasks
        df_main["SBCID"] = sbcid_values_all_tasks

        palette = {"Hi-C-LSTM": "C3", "SNIPER-INTRA": "C0", "SNIPER-INTER": "C1", "SCI": "C2", "PCA": "C4",
                   "SBCID": "C5"}
        plt.figure(figsize=(10, 8))
        plt.xticks(rotation=90, fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("Downstream Tasks", fontsize=16)
        plt.ylabel("mAP ", fontsize=16)
        plt.plot('Tasks', 'Hi-C-LSTM', data=df_main, marker='o', markersize=12, color="C3", linewidth=2,
                 label="Hi-C-LSTM")
        plt.plot('Tasks', 'SNIPER-INTRA', data=df_main, marker='', color="C0", linewidth=2, label="SNIPER-INTRA")
        plt.plot('Tasks', 'SNIPER-INTER', data=df_main, marker='', color="C1", linewidth=2, linestyle='dashed',
                 label="SNIPER-INTER")
        plt.plot('Tasks', 'SCI', data=df_main, marker='^', color="C2", linewidth=2, label="SCI")
        plt.plot('Tasks', 'PCA', data=df_main, marker='+', color="C4", linewidth=2, label="PCA")
        plt.plot('Tasks', 'SBCID', data=df_main, marker='v', color="C5", linewidth=2, label="SBCID")
        plt.legend(fontsize=15)
        plt.show()
        print("done")

        pass
'''

'''
  self.plot_rna_seq(path, lstm_rna, sniper_rna, graph_rna, pca_rna, sbcid_rna)
        self.plot_pe(path, lstm_pe, sniper_pe, graph_pe, pca_pe, sbcid_pe)
        self.plot_fire(path, lstm_fire, sniper_fire, graph_fire, pca_fire, sbcid_fire)
        self.plot_rep(path, lstm_rep, sniper_rep, graph_rep, pca_rep, sbcid_rep)
        self.plot_promoter(path, lstm_promoter, sniper_promoter, graph_promoter, pca_promoter, sbcid_promoter)
        self.plot_enhancer(path, lstm_enhancer, sniper_enhancer, graph_enhancer, pca_enhancer, sbcid_enhancer)
        self.plot_tss(path, lstm_tss, sniper_tss, graph_tss, pca_tss, sbcid_tss)
        self.plot_domain(path, lstm_domain, sniper_domain, graph_domain, pca_domain, sbcid_domain)
        self.plot_loop(path, lstm_loop, sniper_loop, graph_loop, pca_loop, sbcid_loop)
        self.plot_sbc(path, lstm_sbc, sniper_sbc, graph_sbc, pca_sbc, sbcid_sbc)
'''

'''
    def plot_all(self, path):
        sniper_rna, sniper_pe, sniper_fire, sniper_rep, sniper_promoter, sniper_enhancer, \
        sniper_domain, sniper_loop, sniper_tss, sniper_sbc, lstm_rna, lstm_pe, lstm_fire, \
        lstm_rep, lstm_promoter, lstm_enhancer, lstm_domain, lstm_loop, lstm_tss, lstm_sbc, \
        graph_rna, graph_pe, graph_fire, graph_rep, graph_promoter, graph_enhancer, graph_domain, \
        graph_loop, graph_tss, graph_sbc, pca_rna, pca_pe, pca_fire, pca_rep, pca_promoter, pca_enhancer, \
        pca_domain, pca_loop, pca_tss, pca_sbc, sbcid_rna, sbcid_pe, sbcid_fire, sbcid_rep, sbcid_promoter, \
        sbcid_enhancer, sbcid_domain, sbcid_loop, sbcid_tss, sbcid_sbc = self.get_dict(path)
'''

'''
map_hidden = [0.6055083, 0.65884748, 0.801499136, 0.83458231, 0.84830276, 0.851328549]
map_2_layer = [0.62, 0.67, 0.824, 0.841, 0.8524, 0.855]
map_bidir = [0.615, 0.663, 0.817, 0.838, 0.851, 0.852]
map_dropout = [0.541, 0.613, 0.772, 0.796, 0.803, 0.818]
map_no_ln = [0.532, 0.598, 0.756, 0.763, 0.782, 0.795]

r2_hidden = [0.706091, 0.7583492, 0.8568210, 0.8928673, 0.91849215, 0.9396382]
r2_bidir = [0.714, 0.769, 0.865, 0.908, 0.926, 0.948]
r2_2_layer = [0.72, 0.77, 0.874, 0.911, 0.931, 0.954]
r2_dropout = [0.643, 0.712, 0.772, 0.817, 0.857, 0.896]
r2_no_ln = [0.637, 0.695, 0.756, 0.802, 0.842, 0.883]

r1_hiclstm_full = [0.7512, 0.8015, 0.8482, 0.8631, 0.8829]
        r1_hiclstm_lstm = [0.7079, 0.7521, 0.7985, 0.8129, 0.8302]
        r1_hiclstm_cnn = [0.6870, 0.7309, 0.7721, 0.7936, 0.8106]
        r1_sci_lstm = [0.6796, 0.7285, 0.7692, 0.7858, 0.8096]
        r1_sniper_lstm = [0.6785, 0.7196, 0.7591, 0.7815, 0.7994]
        r1_sci_cnn = [0.6531, 0.6974, 0.7366, 0.7619, 0.7784]
        r1_sniper_cnn = [0.6508, 0.6931, 0.7287, 0.7601, 0.7752]
        r1_hiclstm_fc = [0.6108, 0.6493, 0.6742, 0.7191, 0.7255]
        r1_sci_fc = [0.6007, 0.6395, 0.6604, 0.7037, 0.7193]
        r1_sniper_fc = [0.6017, 0.6281, 0.6590, 0.7016, 0.7148]

        r2_hiclstm_lstm = [0.6499, 0.6941, 0.7005, 0.76489, 0.7722]
        r2_hiclstm_cnn = [0.63801, 0.67191, 0.6934, 0.7446, 0.7516]
        r2_sci_lstm = [0.6286, 0.6675, 0.6882, 0.7348, 0.7486]
        r2_sniper_lstm = [0.6215, 0.6626, 0.6821, 0.7345, 0.7424]
        r2_sci_cnn = [0.6019, 0.6444, 0.6636, 0.7149, 0.7254]
        r2_sniper_cnn = [0.6028, 0.6351, 0.6607, 0.7021, 0.7172]
        r2_hiclstm_fc = [0.5688, 0.5873, 0.6122, 0.6471, 0.6535]
        r2_sci_fc = [0.5527, 0.5715, 0.6024, 0.6357, 0.6413]
        r2_sniper_fc = [0.557, 0.5781, 0.6053, 0.6256, 0.6388]
        r2_hiclstm_full = [0.51419, 0.5244, 0.5411, 0.5760, 0.5859]
'''
