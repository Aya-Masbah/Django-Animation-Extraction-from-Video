from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User
from django.views.generic import (
    ListView,
    DetailView,
    CreateView,
    UpdateView,
    DeleteView
)
from .models import Post
from django.urls import reverse_lazy
from django.db.models import Q
import cv2
import mediapipe as mp
import numpy as np
import os
from django.conf import settings
from cvzone.PoseModule import PoseDetector
from django.shortcuts import render, redirect
from django.http import HttpResponse
import threading
from . import global_vars
from .body import BodyThread

from django.shortcuts import render, get_object_or_404
from .models import Post

from django.shortcuts import redirect

def run_mediapipe_script(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    video_path = post.file.path  # Utilisez 'file' à la place de 'video'

    # Démarrer le thread de traitement avec le chemin de la vidéo
    body_thread = BodyThread(video_path)
    body_thread.start()

    # Rediriger vers le détail du post
    return redirect('post-detail', pk=post.pk)

def home(request):
    context = {
        'posts': Post.objects.all()
    }
    return render(request, 'blog/home.html', context)

def search(request):
    template = 'blog/home.html'
    query = request.GET.get('q')
    result = Post.objects.filter(Q(title__icontains=query) | Q(author__username__icontains=query) | Q(content__icontains=query))
    context = {'posts': result}
    return render(request, template, context)

def convert_video_to_animation(video_path, output_path):
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_path_mp4 = output_path.replace('.avi', '.mp4')
    out = cv2.VideoWriter(output_path_mp4, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(frame)
        opImg = np.ones((frame_height, frame_width, 3), np.uint8) * 255
        if results.pose_landmarks:
            mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

        out.write(opImg)

    cap.release()
    out.release()
    return output_path_mp4

def generate_pose_csv(video_path, csv_output_path):
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()
    pose_list = []

    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.findPose(img)
        lm_list, bbox_info = detector.findPosition(img)

        if bbox_info:
            lm_string = ''
            for lm in lm_list:
                lm_string += f'{lm[1]}, {img.shape[0] - lm[2]}, {lm[2]} '
            pose_list.append(lm_string)

    cap.release()

    with open(csv_output_path, 'w') as f:
        f.writelines(["%s\n" % item for item in pose_list])

class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    template_name = 'blog/post_form.html'
    fields = ['title', 'content', 'file']

    def form_valid(self, form):
        form.instance.author = self.request.user
        response = super().form_valid(form)
        
        if form.instance.file:
            video_path = form.instance.file.path
            base_filename = os.path.splitext(os.path.basename(video_path))[0]
            animation_relative_path = os.path.join('animations', f'{base_filename}_animation.avi')
            csv_relative_path = os.path.join('csv', f'{base_filename}_poses.csv')
            animation_path = os.path.join(settings.MEDIA_ROOT, animation_relative_path)
            csv_output_path = os.path.join(settings.MEDIA_ROOT, csv_relative_path)

            os.makedirs(os.path.dirname(animation_path), exist_ok=True)
            os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

            # Generate animation
            animation_path_mp4 = convert_video_to_animation(video_path, animation_path)
            form.instance.animation = animation_relative_path.replace('.avi', '.mp4')

            # Generate CSV
            generate_pose_csv(video_path, csv_output_path)
            form.instance.csv_file = csv_relative_path

            form.instance.save()
        
        return response

class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Post
    template_name = 'blog/post_form.html'
    fields = ['title', 'content', 'file']

    def form_valid(self, form):
        form.instance.author = self.request.user
        response = super().form_valid(form)
        
        if form.instance.file:
            video_path = form.instance.file.path
            base_filename = os.path.splitext(os.path.basename(video_path))[0]
            animation_relative_path = os.path.join('animations', f'{base_filename}_animation.avi')
            csv_relative_path = os.path.join('csv', f'{base_filename}_poses.csv')
            animation_path = os.path.join(settings.MEDIA_ROOT, animation_relative_path)
            csv_output_path = os.path.join(settings.MEDIA_ROOT, csv_relative_path)

            os.makedirs(os.path.dirname(animation_path), exist_ok=True)
            os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

            # Generate animation
            animation_path_mp4 = convert_video_to_animation(video_path, animation_path)
            form.instance.animation = animation_relative_path.replace('.avi', '.mp4')

            # Generate CSV
            generate_pose_csv(video_path, csv_output_path)
            form.instance.csv_file = csv_relative_path

            form.instance.save()
        
        return response

    def test_func(self):
        post = self.get_object()
        return self.request.user == post.author

class PostListView(ListView):
    model = Post
    template_name = 'blog/home.html'
    context_object_name = 'posts'
    ordering = ['-date_posted']
    paginate_by = 2

class UserPostListView(ListView):
    model = Post
    template_name = 'blog/user_posts.html'
    context_object_name = 'posts'
    paginate_by = 2

    def get_queryset(self):
        user = get_object_or_404(User, username=self.kwargs.get('username'))
        return Post.objects.filter(author=user).order_by('-date_posted')

class PostDetailView(DetailView):
    model = Post
    template_name = 'blog/post_detail.html'

class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    success_url = '/'
    template_name = 'blog/post_confirm_delete.html'

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False

def about(request):
    return render(request, 'blog/about.html', {'title': 'About'})
