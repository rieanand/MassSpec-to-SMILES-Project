'''Code responsible fro executing the entire pipeline.'''

from model_service import ModelService

def main():
    ml_svc = ModelService()
    ml_svc.load_model('name')
    pred = ml_svc.predict([])
    print(pred)

if __name__ == '__main__':
    main()