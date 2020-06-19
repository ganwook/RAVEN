import copy
import cv2

from const import *
from sampling import sample_rules
from rendering import render_panel
from build_tree import build_distribute_four

def set_target_with_condition(AoT, attr, value):
    results = {}
    Components = AoT.children[0].children
    for comp_idx, comp in enumerate(Components):
        Entities = comp.children[0].children
        entity_idxs = []
        for ent_idx, ent in enumerate(Entities):
            if attr == "Type":
                if ent.type.get_value() == value:
                    entity_idxs += [ent_idx]
            if attr == "Size":
                if ent.size.get_value() == value:
                    entity_idxs += [ent_idx]
            if attr == "Color":
                if ent.color.get_value() == value:
                    entity_idxs += [ent_idx]
        results[comp_idx] = entity_idxs
        
    return results

def generate_pairs(n_oper, pair_per_oper, config_attr, debug=False):
    dataset = {}
    rule_configs = {}
    
    # for each operation
    for i in range(n_oper):
        # sample only the unique operation
        while True:
            rule_groups = sample_rules()
            rule_config = [str(rule_groups[0].__dict__[attr]) for attr in config_attr]
            rule_config_str = ' '.join(rule_config)
            if rule_config_str not in list(rule_configs.values()):
                rule_configs[i] = rule_config_str
                break
        
        # sample an image conditioned on the operation
        rule = rule_groups[0]
        lst_pairs = []
        for j in range(pair_per_oper):
            root = build_distribute_four() # structure_config
            
            # Sample until getting non-error case
            start_node = None
            while start_node is None:
                try:
                    start_node = root.sample(rule)
                    tgt_idxs = set_target_with_condition(start_node, rule.tgt_attr, rule.tgt_value)
                    if len(tgt_idxs[0])==0:
                        start_node = None 
                except:
                    break
            if start_node is None:
                continue
                
            output = rule.apply_rule(start_node, entity_idxs = tgt_idxs[0])
            
            try:
                lst_pairs += [{ 'in'  : render_panel(start_node),
                                'out' : render_panel(output)}]
            except:
                if debug:
                    print('Iteration does not completed. Return an error case')
                    return output
                else:
                    print(2)
                    continue

        dataset[i] = {'rule' : rule_config_str,
                      'data' : lst_pairs    }
        if debug:
            print(f'operation {i} : {len(lst_pairs)}')
    if debug:
        print(f'#(operations) : {len(dataset)}')
        print('Iteration ends')

    return dataset
            



    