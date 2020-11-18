def adj_find_target_str_bbg_data(df, target_str='date'):
    if df.columns[0] != target_str:
        df.columns = df.iloc[0]
        while df.columns[0] != target_str:
            df.drop(df.index[0], inplace=True)
            df.columns = df.iloc[0]
        df.drop(df.index[0], inplace=True)

    return df


def merge_data(df_list, freq='D'):
    list_out = []
    for i, df in enumerate(df_list):
        # print(i)
        df_loop = df.resample(freq).mean()
        list_out.append(df_loop)
    df_out = pd.concat(list_out, axis=1)

    return df_out


def make_df_from_path(path, threshold_to_exclude=365):
    list_out = []
    excluded_list = []
    for file in os.listdir(path):
        df_loop = pd.read_csv(os.path.join(path, file))
        if len(df_loop) >= threshold_to_exclude:
            df_loop = adj_find_target_str_bbg_data(df_loop)
            names_in_col = [df_loop.columns[0]] + [file.replace('.csv', '')]
            df_loop.columns = names_in_col

            df_loop[names_in_col[0]] = pd.to_datetime(df_loop[names_in_col[0]])
            df_loop[names_in_col[1]] = df_loop[names_in_col[1]].astype(float)

            df_loop = df_loop.set_index(names_in_col[0])
            list_out.append(df_loop)
        else:
            excluded_list.append(file)
    df_out = merge_data(list_out)
    return df_out, excluded_list