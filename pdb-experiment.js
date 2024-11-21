module.exports = {
    apps: [
      {
        name: "hyperparameter-search",                          // Process name
        script: "/home/paperspace/sergio/folding/folding/experiments/hyperparameter-search.py",
        interpreter: "python3",                                 // Specify Python interpreter
        autorestart: true,                                      // Restart on crash
        watch: false,                                           // No file watching                             // Optional: Restart if memory exceeds 1 GB
        env_production: {                                       // Optional: Environment variables for production
          NODE_ENV: "production"
        }
      }
    ]
  };