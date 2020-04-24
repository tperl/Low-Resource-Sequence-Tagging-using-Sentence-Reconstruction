import os
import subprocess
import time

def main():
    lang = 'dutch'
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    lang_list = ['dutch','spanish','english']
    model_list = ['baseline','transfer']
    reconstruction_list = [False, True]
    for model_type in model_list:
        for lang in lang_list:
            for i in range(0,3):
                for recon in reconstruction_list:
                    if recon:
                        log_path = './logs/' + lang + '_' + model_type + '_L2_' + str(i) + '.txt'
                    else:
                        log_path = './logs/' + lang + '_' + model_type + '_' + str(i) + '.txt'
                    if lang == 'english':
                        src_lang = 'dutch'
                    else:
                        src_lang = 'english'

                    if recon:
                        cmd = 'python train.py --lang=' + lang + ' --src_lang=' + src_lang + ' --model=' + model_type + ' --add_reconstruction=True --learning_rate=0.01 --verbosity=0 --num_of_epochs=60'
                    else:
                        cmd = 'python train.py --lang=' + lang + ' --src_lang=' + src_lang + ' --model=' + model_type + ' --learning_rate=0.01 --verbosity=0 --num_of_epochs=60'

                    os.system(cmd)
                    if recon:
                        cmd = 'python test.py --lang='+lang+' --model='+model_type+' --add_reconstruction=True --log_results='+log_path
                    else:
                        cmd = 'python test.py --lang=' + lang + ' --model=' + model_type + ' --log_results=' + log_path
                    os.system(cmd)



    print('done')
if __name__ == '__main__':
    main()