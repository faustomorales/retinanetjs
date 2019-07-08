FROM node:8.11.0
WORKDIR /usr/src
COPY ./package.json ./package.json
COPY ./yarn.lock ./yarn.lock
RUN yarn install
RUN curl https://sdk.cloud.google.com | bash
ENV PATH=/root/google-cloud-sdk/bin:$PATH