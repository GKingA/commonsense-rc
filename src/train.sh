python main.py --model race-default.mdl --epoch 10 --four_lang False --race True >> commonsense_default_race.txt
python main.py --model 1000-default.mdl --pretrained ./checkpoint/race-default.mdl --seed 1000 --four_lang False --test_mode True >> commonsense_default_1.txt
python main.py --model 2000-default.mdl --pretrained ./checkpoint/race-default.mdl --seed 2000 --four_lang False --test_mode True >> commonsense_default_2.txt
python main.py --model 3000-default.mdl --pretrained ./checkpoint/race-default.mdl --seed 3000 --four_lang False --test_mode True >> commonsense_default_3.txt
python main.py --model 4000-default.mdl --pretrained ./checkpoint/race-default.mdl --seed 4000 --four_lang False --test_mode True >> commonsense_default_4.txt
python main.py --model 5000-default.mdl --pretrained ./checkpoint/race-default.mdl --seed 5000 --four_lang False --test_mode True >> commonsense_default_5.txt
python main.py --model 6000-default.mdl --pretrained ./checkpoint/race-default.mdl --seed 6000 --four_lang False --test_mode True >> commonsense_default_6.txt
python main.py --model 7000-default.mdl --pretrained ./checkpoint/race-default.mdl --seed 7000 --four_lang False --test_mode True >> commonsense_default_7.txt
python main.py --model 8000-default.mdl --pretrained ./checkpoint/race-default.mdl --seed 8000 --four_lang False --test_mode True >> commonsense_default_8.txt
python main.py --model 9000-default.mdl --pretrained ./checkpoint/race-default.mdl --seed 9000 --four_lang False --test_mode True >> commonsense_default_9.txt

