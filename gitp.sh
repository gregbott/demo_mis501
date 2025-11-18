#!/bin/bash

# Git add, commit, push, and status script

echo "Adding all changes..."
git add .

echo "Committing changes..."
git commit -m 'via script'

echo "Pushing to remote..."
git push

echo "Current status:"
git status
