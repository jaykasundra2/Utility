doc = {'que':[{'name':'personal_info',
               'address':[{'label':'city',
                           'value':'mumbai'},
                          {'label':'state',
                           'value':'maharashtra'
                           }
                          ]
               },
              {'name':'company_details',
               'sector_details':[{'label':'sector',
                                  'value':'agriculture'},
                                 {'label':'sub-sector',
                                  'value':'paddy'}
                                 ],
               'products':[{'name':'chickpea','quantity':1000,'price':120},
                           {'name':'mustard','quantity':1500,'price':80}
                           ]
               }
            ],
       'created_at':{'year':'2020','month':"02",'day':'02'}
             }

def get_element(obj,key,default = ''):
    if isinstance(obj, list):
        #print("returning value from a list")
        return [v.get(key, default) if v else None for v in obj]
    if isinstance(obj, dict):
        #print("returning value from a dict")
        return obj.get(key)
def get_element_from_list(obj,key):
    return [elem for elem in obj if key in elem.values()][0]

def my_deep_get(doc,keys,default=''):
    try:
        path_list = path.split(".")
        end_loop = False; 
        sub_doc = doc; 
        counter=0
        while not end_loop:
            #print(counter)
            #print(path_list[counter])
            if path_list[counter]=='check_values_pick_dict':
                counter+=1
                sub_doc = [elem for elem in sub_doc if path_list[counter] in elem.values()][0]
                #print(sub_doc)
            else:
                sub_doc = get_element(sub_doc,path_list[counter])
                #print(sub_doc)
            
            counter+=1
            if counter>=len(path_list):
                end_loop=True
                return(sub_doc)
    except:
        print("Error!!!!!!!")
        return default
path = 'que.check_values_pick_dict.personal_info.address.check_values_pick_dict.city.value'
path = 'que.check_values_pick_dict.company_details.products.name'
path = 'created_at.year'
path = 'que.check_values_pick_dict.company_details.sector_details.check_values_pick_dict.sector.value'
my_deep_get(doc,path)
