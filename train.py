import numpy as np

def train_cql(model, env, dataset, n_steps):
    # Setup evaluators
    from d3rlpy.metrics import TDErrorEvaluator, EnvironmentEvaluator

    td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)
    env_evaluator = EnvironmentEvaluator(env)

    # Start training
    model.fit(
        dataset,
        n_steps=n_steps,
        evaluators={
            'td_error': td_error_evaluator,
            'environment': env_evaluator,
        },
    )

    # After training, let's test the model
    observation, _ = env.reset()
    
    # Return actions based on the greedy-policy
    action = model.predict(np.expand_dims(observation, axis=0))
    
    # Estimate action-values
    value = model.predict_value(np.expand_dims(observation, axis=0), action)

    print(f"Sample prediction - Action: {action}, Value: {value}")

    # Save the model
    model.save('cql_model.d3')
    model.save_model('cql_model.pt')
    model.save_policy('cql_policy.pt')
    model.save_policy('cql_policy.onnx')

    return model

# You might want to add a function to load the model if needed
def load_cql_model(path):
    import d3rlpy
    return d3rlpy.load_learnable(path)