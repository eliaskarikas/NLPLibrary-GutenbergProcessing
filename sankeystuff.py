import plotly.graph_objects as go

def code_mapping(df, src, targ):
    """ map labels in src and targ columns to integers """
    labels = list(df[src]) + list(df[targ])
    labels = (list(set(labels)))

    #print(labels)

    codes = list(range(len(labels)))
    #print(codes)

    lcmap = dict(zip(labels, codes))
    #print(lcmap)

    df = df.replace({src: lcmap, targ: lcmap})
    #print(df)

    return df, labels

def make_sankey(df, src, targ, vals):
    df, labels = code_mapping(df, src, targ)

    link = {'source':df[src], 'target':df[targ], 'value':df[vals]}

    node = {'pad':100, 'thickness':10,
            'line':{'color':'black', 'width':2},
            'label':labels}

    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)
    fig.show()
