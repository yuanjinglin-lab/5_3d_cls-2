
data_root_path = Path('./data')
data_path = data_root_path / 'pkl'

all_data_list = list(data_path.glob('*.pkl'))

kf = KFold(n_splits=3,shuffle=True, random_state=3047)

split_data ={}
for fold_index, (train_index, validation_index) in enumerate(kf.split(all_data_list)):
    train_set =[str(all_data_list[i].absolute()) for i in train_index]
    validation_set =[str(all_data_list[i])for i in validation_index]


    dict_name = f'fold_{fold_index}'
    # 将当前折的数据保存到字典中
    split_data[dict_name] = {
        'train': train_set,
        'validation':validation_set
    }
with open('split result.ison','w')as file:
    json.dump(split_data, file, indent=4)
print("Data has been successfully split and written to 'split_result.ison'.")