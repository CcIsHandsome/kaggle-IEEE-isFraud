import pandas as pd

sub_9467 = pd.read_csv('./submission_9467.csv')
sub_9473 = pd.read_csv('./submission_9473.csv')
sub_9477 = pd.read_csv('./submission_9477.csv')
sub_9487 = pd.read_csv('./sub_blend_9487.csv')
sub_9498 = pd.read_csv('./stack_gmean_9498.csv')
sub_9518 = pd.read_csv('./submission_9518.csv')
sub_9518_2 = pd.read_csv('./submission _9518_2.csv')
sub_9519 = pd.read_csv('./submission_9519.csv')
sub_9524 = pd.read_csv('./submission_9524.csv')
sub_9492_single = pd.read_csv('./submission_9492_single.csv')

sub_blend = pd.read_csv('../input/sample_submission.csv')
ans = 0.1 * sub_9473['isFraud'] + 0.5 * sub_9524['isFraud'] + 0.4 * sub_9492_single['isFraud']
# ans = 0.5 * sub_9473['isFraud'] + 0.5 * sub_9492_single['isFraud']
sub_blend['isFraud'] = ans
sub_blend.to_csv('./sub_blend.csv', index=False)
