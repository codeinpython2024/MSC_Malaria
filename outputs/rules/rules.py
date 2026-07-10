def findDecision(obj): #obj[0]: b19, obj[1]: b4, obj[2]: v106, obj[3]: v190, obj[4]: v119, obj[5]: v158, obj[6]: v127, obj[7]: v128, obj[8]: v129, obj[9]: v113, obj[10]: v116, obj[11]: hml20
   # {"feature": "b19", "instances": 316, "metric_value": 1.0, "depth": 1}
   if obj[0]>6:
      # {"feature": "v158", "instances": 311, "metric_value": 0.9998, "depth": 2}
      if obj[5] == '0':
         # {"feature": "hml20", "instances": 280, "metric_value": 0.9928, "depth": 3}
         if obj[11] == '0':
            # {"feature": "v113", "instances": 260, "metric_value": 0.9829, "depth": 4}
            if obj[9] == '31':
               # {"feature": "v129", "instances": 138, "metric_value": 0.817, "depth": 5}
               if obj[8] == '31':
                  return 'Positive'
               elif obj[8] == '13':
                  return 'Negative'
               elif obj[8] == '22':
                  return 'Negative'
               elif obj[8] == '12':
                  return 'Negative'
               else:
                  return 'Positive'
            elif obj[9] == '43':
               # {"feature": "v106", "instances": 56, "metric_value": 0.9963, "depth": 5}
               if obj[2] == '0':
                  return 'Positive'
               elif obj[2] == '2':
                  return 'Negative'
               elif obj[2] == '1':
                  return 'Negative'
               else:
                  return 'Negative'
            elif obj[9] == '21':
               # {"feature": "v128", "instances": 44, "metric_value": 0.9624, "depth": 5}
               if obj[7] == '21':
                  return 'Positive'
               elif obj[7] == '31':
                  return 'Negative'
               elif obj[7] == '11':
                  return 'Negative'
               else:
                  return 'Negative'
            elif obj[9] == '32':
               # {"feature": "v128", "instances": 22, "metric_value": 0.684, "depth": 5}
               if obj[7] == '31':
                  return 'Negative'
               elif obj[7] == '21':
                  return 'Positive'
               elif obj[7] == '22':
                  return 'Negative'
               else:
                  return 'Negative'
            else:
               return 'Positive'
         elif obj[11] == '1':
            # {"feature": "v119", "instances": 20, "metric_value": 0.7219, "depth": 4}
            if obj[4] == '0':
               # {"feature": "v106", "instances": 18, "metric_value": 0.65, "depth": 5}
               if obj[2] == '2':
                  return 'Negative'
               elif obj[2] == '0':
                  return 'Negative'
               elif obj[2] == '1':
                  return 'Negative'
               elif obj[2] == '3':
                  return 'Negative'
               else:
                  return 'Negative'
            elif obj[4] == '1':
               # {"feature": "v190", "instances": 2, "metric_value": 1.0, "depth": 5}
               if obj[3] == '2':
                  return 'Positive'
               elif obj[3] == '4':
                  return 'Negative'
               else:
                  return 'Positive'
            else:
               return 'Negative'
         else:
            return 'Positive'
      elif obj[5] == '2':
         # {"feature": "v128", "instances": 23, "metric_value": 0.6666, "depth": 3}
         if obj[7] == '31':
            # {"feature": "v113", "instances": 19, "metric_value": 0.6292, "depth": 4}
            if obj[9] == '43':
               return 'Negative'
            elif obj[9] == '31':
               # {"feature": "v116", "instances": 6, "metric_value": 0.65, "depth": 5}
               if obj[10] == '22':
                  return 'Negative'
               elif obj[10] == '12':
                  return 'Negative'
               else:
                  return 'Negative'
            elif obj[9] == '21':
               # {"feature": "b4", "instances": 2, "metric_value": 1.0, "depth": 5}
               if obj[1] == '1':
                  return 'Negative'
               else:
                  return 'Negative'
            elif obj[9] == '32':
               # {"feature": "b4", "instances": 2, "metric_value": 1.0, "depth": 5}
               if obj[1] == '1':
                  return 'Negative'
               elif obj[1] == '2':
                  return 'Positive'
               else:
                  return 'Negative'
            else:
               return 'Negative'
         elif obj[7] == '21':
            return 'Negative'
         elif obj[7] == '13':
            return 'Positive'
         elif obj[7] == '22':
            return 'Negative'
         else:
            return 'Negative'
      elif obj[5] == '1':
         return 'Negative'
      else:
         return 'Positive'
   elif obj[0]<=6:
      return 'Negative'
   else:
      return 'Positive'
