import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



bleu_df = pd.DataFrame(data={'BLEU Scores': ['Baseline BLEU','Fine-tuned BlEU']*7, \
    'Languages': ['Filipino', 'Filipino', 'Hindi', 'Hindi', 'Indonesian', \
        'Indonesian', 'Malay', 'Malay', 'Burmese', 'Burmese', 'Thai', 'Thai',\
            'Vietnamese', 'Vietnamese'], \
                'BLEU Score Zh': [24.45, 31.81, 30.49, 32.43, 26.41, 29.47,\
                    31.65, 33.35, 19.82, 26.81, 20.42, 23.91, 32.14, 36.91],\
                        'BLEU Score Ja': [17.62, 28.37, 17.96, 29.63, 26.33, 29.54,\
                            31.74, 39.88, 4.84, 25.11, 17.21, 21.42, 25.87, 33.45]\
                                })


sns.factorplot(x = 'Languages', y='BLEU Score Zh', hue = 'BLEU Scores',data=bleu_df, kind='bar')
sns.factorplot(x = 'Languages', y='BLEU Score Ja', hue = 'BLEU Scores',data=bleu_df, kind='bar')

plt.show()