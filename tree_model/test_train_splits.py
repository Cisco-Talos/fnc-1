import pandas as pd
import numpy as np
from collections import Counter
import random as rn
import math as mt

def split_it( df, percentage ):
    num_items = sum( df.counts )
    num_rows = len( df )

    target_size = mt.floor(num_items*percentage)
    indices = {}
    tagged_rows = 0
    while tagged_rows < target_size:
        indx = rn.randrange( num_rows )
        if indx in indices:
            pass
        else:
            indices[indx] = df.counts[indx]
            tagged_rows += indices[indx]

    a_indx = [ True if indx in indices.keys() else False for indx in df.index ]
    b_indx = [ not a for a in a_indx]

    return ( df.key[a_indx].reset_index(drop=True), df.key[b_indx].reset_index(drop=True) )

def break_it( df, percentage ):
    ''' break_it( df, percentage )
        df : data frame with 'Headline', and 'Body ID' cols
        percentage : targe percentage to split into
        output: df broken into non-overlapping dataframes, as close to percentage split as possible
        eg: (df_a, df_b) = break_it( df_related, .2)
    '''
    num_rows = len( df )
    target_size = mt.floor(num_rows*percentage)
    tagged_rows = 0

    found_rows = pd.Series( [False] * num_rows )
    while tagged_rows < target_size:

        indx = rn.randrange( num_rows )

        this_headline = df['Headline'][indx]
        this_body_id = df['Body ID'][indx]

        this_indx = ( df['Headline'] == this_headline ) | ( df['Body ID'] == this_body_id );

        c = Counter( this_indx )
        this_true_count = c[True]
        last_true_count = 0

        while last_true_count != this_true_count:
            last_true_count = this_true_count
            this_indx = find_all( df, this_indx )
            c = Counter( this_indx )
            this_true_count = c[True]

            print "true count {}\n".format( this_true_count )

        found_rows = found_rows | this_indx
        tagged_rows += this_true_count

    return (df[found_rows], df[~found_rows] )

def find_all( df, this_indx ):

    this_headlines = df['Headline'][this_indx]
    this_body_id   = df['Body ID'][this_indx]


    df_h_indx  = [ True if k in this_headlines.values else False for k in df['Headline'] ]
    df_b_indx  = [ True if k in this_body_id.values else False for k in df['Body ID'] ]


    return this_indx | pd.Series(df_h_indx) | pd.Series(df_b_indx)



def get_repeats_df(df, groupby_cols):

    agg_functions = {
        stance: (lambda x: Counter(x)),
        body: 'count'
    }

    renaming = {
        stance: 'stances',
        body: 'counts'
    }

    grouped =  df.groupby(groupby_cols, as_index=False)
    aggregated = grouped.agg(agg_functions).rename(columns = renaming).sort_values('counts', ascending=False)
    return aggregated



body_id = 'Body ID'
stance = 'Stance'
headline = 'Headline'
body = 'articleBody'

stances = pd.read_csv('train_stances.csv', encoding='utf-8')
bodies = pd.read_csv('train_bodies.csv', encoding='utf-8')

df = stances.merge(bodies,on='Body ID')

df.apply(lambda row: 'unrelated' if row[stance] == 'unrelated' else 'related', axis=1).value_counts()
is_related = 'is_related'
df[is_related] = df.apply(lambda row: row[stance] != 'unrelated', axis=1)

# only keeping the related for now ... more interesting of the classification topic
df_related = df[df.is_related].drop_duplicates().reset_index(drop=True)

run_headline_body_split = False
run_no_overlap_split = True

df_related = df_related.drop(body, axis=1)
if run_headline_body_split:
    headline_related_repeats = get_repeats_df(df_related, headline).rename( columns = { headline : 'key'} )
    body_related_repeats = get_repeats_df(df_related, body_id).rename( columns = { body_id : 'key' } )

    for ii in range( 0, 5 ):

        (test_h,train_h) = split_it( headline_related_repeats, .2 )
        (test_b,train_b) = split_it( body_related_repeats, .2 )

        df_test_h_indx  = [ True if k in test_h.values else False for k in df_related[headline] ]
        df_train_h_indx = [ True if not a else False for a in df_test_h_indx ]

        df_test_b_indx  = [ True if k in test_b.values else False for k in df_related[body_id] ]
        df_train_b_indx = [ True if not a else False for a in df_test_b_indx ]

        df_related[df_test_h_indx].to_csv( 'test_h_{}.csv'.format( ii ), index=False, encoding='utf-8' )
        df_related[df_train_h_indx].to_csv( 'train_h_{}.csv'.format( ii ), index=False, encoding='utf-8' )

        df_related[df_test_b_indx].to_csv( 'test_b_{}.csv'.format( ii ), index=False, encoding='utf-8' )
        df_related[df_train_b_indx].to_csv( 'train_b_{}.csv'.format( ii ), index=False, encoding='utf-8' )

if run_no_overlap_split:
    for ii in range( 0, 5 ):
        (df_test, df_train ) = break_it( df_related, .2 )

        # labeled d for disjoint
        df_test.to_csv( 'test_d_{}.csv'.format( ii ), index=False, encoding='utf-8' )
        df_train.to_csv( 'train_d_{}.csv'.format( ii ), index=False, encoding='utf-8' )

 #   Copyright 2017 Cisco Systems, Inc.
 #  
 #   Licensed under the Apache License, Version 2.0 (the "License");
 #   you may not use this file except in compliance with the License.
 #   You may obtain a copy of the License at
 #  
 #     http://www.apache.org/licenses/LICENSE-2.0
 #  
 #   Unless required by applicable law or agreed to in writing, software
 #   distributed under the License is distributed on an "AS IS" BASIS,
 #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 #   See the License for the specific language governing permissions and
 #   limitations under the License.
